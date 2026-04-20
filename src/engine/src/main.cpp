#include <engine/scheduler.hpp>
#include <engine/memory_graph.hpp>
#include <slab/slab_allocator.hpp>
#include <metric/fisher_rao.hpp>
#include <sheaf/coboundary.hpp>
#include <langevin/sde_stepper.hpp>
#include <persist/sqlite_store.hpp>

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Global scheduler pointer for signal handler
static slm::engine::Scheduler* g_scheduler = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_scheduler) {
        g_scheduler->request_stop();
    }
}

int main(int argc, char* argv[]) {
    // Default shm_path: ~/.slmfs/ipc_shm.bin (file-backed mmap)
    std::filesystem::path shm_path;
    std::filesystem::path db_path = ".slmfs/memory.db";
    uint32_t shm_size = 4 * 1024 * 1024;
    uint32_t slab_size = 64 * 1024;
    uint32_t ctrl_size = 4096;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.starts_with("--shm-path=")) {
            shm_path = arg.substr(11);
        } else if (arg.starts_with("--db-path=")) {
            db_path = arg.substr(10);
        }
    }

    // Default shm_path under ~/.slmfs/
    if (shm_path.empty()) {
        const char* home = std::getenv("HOME");
        if (!home) {
            std::cerr << "HOME not set\n";
            return 1;
        }
        shm_path = std::filesystem::path(home) / ".slmfs" / "ipc_shm.bin";
    }

    if (!db_path.parent_path().empty()) {
        std::filesystem::create_directories(db_path.parent_path());
    }
    if (!shm_path.parent_path().empty()) {
        std::filesystem::create_directories(shm_path.parent_path());
    }

    uint32_t slab_count = (shm_size - ctrl_size) / slab_size;

    std::cout << "SLMFS Engine starting...\n"
              << "  shm_path:   " << shm_path << "\n"
              << "  db_path:    " << db_path << "\n"
              << "  slab_count: " << slab_count << "\n"
              << "  slab_size:  " << slab_size << "\n";

    // File-backed mmap: create/open a regular file instead of POSIX shm
    int shm_fd = open(shm_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) {
        std::cerr << "Failed to open shm file: " << shm_path << "\n";
        return 1;
    }

    if (ftruncate(shm_fd, shm_size) < 0) {
        std::cerr << "Failed to resize shm file\n";
        close(shm_fd);
        return 1;
    }

    void* shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to mmap shm file\n";
        close(shm_fd);
        return 1;
    }

    // Zero the entire region to clear stale queue state from previous runs.
    // File-backed mmap persists across restarts, so without this the SPSC
    // ring buffer head/tail and buffered handles would carry over.
    std::memset(shm_ptr, 0, shm_size);

    slm::slab::SlabAllocator slab(shm_ptr, slab_count, slab_size, ctrl_size);
    slm::engine::MemoryGraph graph;
    slm::metric::FisherRaoMetric metric;
    slm::sheaf::CoboundaryOperator sheaf;
    slm::langevin::LangevinStepper langevin({
        .dt = 5.0f,
        .lambda_decay = 0.01f,
        .noise_scale = 0.001f,
        .archive_threshold = 0.95f,
    });
    slm::persist::SqliteStore store(db_path);

    store.load(graph);
    std::cout << "Loaded " << graph.size() << " nodes from " << db_path << "\n";

    slm::engine::Scheduler scheduler(
        slab, slab.cmd_queue(), graph, metric, sheaf, langevin, store, {}
    );

    g_scheduler = &scheduler;
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);

    std::thread engine_thread([&] {
        scheduler.run();
    });

    std::cout << "Engine running. PID=" << getpid() << "\n";

    engine_thread.join();

    munmap(shm_ptr, shm_size);
    close(shm_fd);

    std::cout << "Engine stopped.\n";
    return 0;
}
