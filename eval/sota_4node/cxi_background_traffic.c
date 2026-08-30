#define _POSIX_C_SOURCE 200809L

/*
 * Controlled four-node CXI background traffic for Elastic-PD experiments.
 *
 * Launch exactly four ranks per node with block placement.  Traffic can be
 * pairwise full duplex, two-producer/two-decoder fan-in, or three-producer/
 * one-decoder incast.  Matching local ranks retain one flow per Cassini rail
 * when MPICH_OFI_NIC_POLICY maps the four local ranks round-robin.  The
 * program reports both sent and received bytes per node so a claimed fabric
 * bottleneck has an achieved-rate and topology receipt.
 */

#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    EXPECTED_WORLD_SIZE = 16,
    RANKS_PER_NODE = 4,
    WARMUP_ROUNDS = 2,
    MAX_PEERS = 3,
    MAX_INFLIGHT = 8,
    NODE_COUNT = 4,
    TAG_BATCH_STRIDE = NODE_COUNT * MAX_INFLIGHT,
};

enum traffic_pattern {
    PATTERN_PAIRWISE_BIDIR = 0,
    PATTERN_PD_2P2D_INCAST = 1,
    PATTERN_PD_3P1D_INCAST = 2,
};

struct config {
    double duration_s;
    size_t message_bytes;
    int inflight;
    double duty_cycle;
    enum traffic_pattern pattern;
    const char *ready_file;
    const char *start_file;
    const char *stop_file;
};

static void usage(const char *program, int rank) {
    if (rank == 0) {
        fprintf(stderr,
                "usage: %s --duration-s SEC --message-bytes BYTES "
                "--inflight N --duty-cycle FRACTION --pattern "
                "pairwise-bidir|pd-2p2d-incast|pd-3p1d-incast "
                "[--ready-file ABSOLUTE] [--start-file ABSOLUTE] "
                "[--stop-file ABSOLUTE]\n",
                program);
    }
}

static void abort_with(const char *message, int rank) {
    fprintf(stderr, "rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static long long parse_ll(const char *text, const char *name, int rank) {
    char *end = NULL;
    errno = 0;
    long long value = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        char buffer[192];
        snprintf(buffer, sizeof(buffer), "%s must be an integer", name);
        abort_with(buffer, rank);
    }
    return value;
}

static double parse_double(const char *text, const char *name, int rank) {
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value)) {
        char buffer[192];
        snprintf(buffer, sizeof(buffer), "%s must be finite", name);
        abort_with(buffer, rank);
    }
    return value;
}

static enum traffic_pattern parse_pattern(const char *text, int rank) {
    if (strcmp(text, "pairwise-bidir") == 0) {
        return PATTERN_PAIRWISE_BIDIR;
    }
    if (strcmp(text, "pd-2p2d-incast") == 0) {
        return PATTERN_PD_2P2D_INCAST;
    }
    if (strcmp(text, "pd-3p1d-incast") == 0) {
        return PATTERN_PD_3P1D_INCAST;
    }
    abort_with("--pattern is invalid", rank);
    return PATTERN_PAIRWISE_BIDIR;
}

static const char *pattern_name(enum traffic_pattern pattern) {
    switch (pattern) {
    case PATTERN_PAIRWISE_BIDIR:
        return "pairwise-bidir";
    case PATTERN_PD_2P2D_INCAST:
        return "pd-2p2d-incast";
    case PATTERN_PD_3P1D_INCAST:
        return "pd-3p1d-incast";
    }
    return "invalid";
}

static struct config parse_config(int argc, char **argv, int rank) {
    struct config result = {
        .duration_s = -1.0,
        .message_bytes = 0,
        .inflight = 0,
        .duty_cycle = -1.0,
        .pattern = PATTERN_PAIRWISE_BIDIR,
        .ready_file = NULL,
        .start_file = NULL,
        .stop_file = NULL,
    };
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            usage(argv[0], rank);
            abort_with("missing option value", rank);
        }
        const char *name = argv[index];
        const char *value = argv[index + 1];
        if (strcmp(name, "--duration-s") == 0) {
            result.duration_s = parse_double(value, name, rank);
        } else if (strcmp(name, "--message-bytes") == 0) {
            long long parsed = parse_ll(value, name, rank);
            if (parsed > 0) {
                result.message_bytes = (size_t)parsed;
            }
        } else if (strcmp(name, "--inflight") == 0) {
            result.inflight = (int)parse_ll(value, name, rank);
        } else if (strcmp(name, "--duty-cycle") == 0) {
            result.duty_cycle = parse_double(value, name, rank);
        } else if (strcmp(name, "--pattern") == 0) {
            result.pattern = parse_pattern(value, rank);
        } else if (strcmp(name, "--ready-file") == 0) {
            result.ready_file = value;
        } else if (strcmp(name, "--start-file") == 0) {
            result.start_file = value;
        } else if (strcmp(name, "--stop-file") == 0) {
            result.stop_file = value;
        } else {
            usage(argv[0], rank);
            abort_with("unknown option", rank);
        }
    }
    const size_t mib = 1024U * 1024U;
    if (result.duration_s < 1.0 || result.duration_s > 10800.0) {
        abort_with("--duration-s must be in [1, 10800]", rank);
    }
    if (result.message_bytes < mib || result.message_bytes > 64U * mib) {
        abort_with("--message-bytes must be in [1 MiB, 64 MiB]", rank);
    }
    if (result.inflight < 1 || result.inflight > MAX_INFLIGHT) {
        abort_with("--inflight must be in [1, 8]", rank);
    }
    if (result.message_bytes > (size_t)INT32_MAX) {
        abort_with("--message-bytes exceeds MPI int count", rank);
    }
    int max_recv_peers = 1;
    if (result.pattern == PATTERN_PD_2P2D_INCAST) {
        max_recv_peers = 2;
    } else if (result.pattern == PATTERN_PD_3P1D_INCAST) {
        max_recv_peers = 3;
    }
    if (result.message_bytes * (size_t)result.inflight
        * (size_t)max_recv_peers > 512U * mib) {
        abort_with("receive allocation exceeds 512 MiB per rank", rank);
    }
    if (result.duty_cycle < 0.05 || result.duty_cycle > 1.0) {
        abort_with("--duty-cycle must be in [0.05, 1.0]", rank);
    }
    if (result.ready_file != NULL && result.ready_file[0] != '/') {
        abort_with("--ready-file must be absolute", rank);
    }
    if (result.start_file != NULL && result.start_file[0] != '/') {
        abort_with("--start-file must be absolute", rank);
    }
    if (result.stop_file != NULL && result.stop_file[0] != '/') {
        abort_with("--stop-file must be absolute", rank);
    }
    if (result.ready_file != NULL && result.stop_file != NULL
        && strcmp(result.ready_file, result.stop_file) == 0) {
        abort_with("ready and stop files must differ", rank);
    }
    if (result.ready_file != NULL && result.start_file != NULL
        && strcmp(result.ready_file, result.start_file) == 0) {
        abort_with("ready and start files must differ", rank);
    }
    if (result.start_file != NULL && result.stop_file != NULL
        && strcmp(result.start_file, result.stop_file) == 0) {
        abort_with("start and stop files must differ", rank);
    }
    return result;
}

static double seconds_between(const struct timespec *left,
                              const struct timespec *right) {
    return (double)(right->tv_sec - left->tv_sec)
           + (double)(right->tv_nsec - left->tv_nsec) / 1.0e9;
}

static void throttle(double busy_s, double duty_cycle) {
    if (duty_cycle >= 1.0 || busy_s <= 0.0) {
        return;
    }
    double sleep_s = busy_s * (1.0 / duty_cycle - 1.0);
    struct timespec delay = {
        .tv_sec = (time_t)sleep_s,
        .tv_nsec = (long)((sleep_s - floor(sleep_s)) * 1.0e9),
    };
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {
    }
}

struct peer_set {
    int send_count;
    int send[MAX_PEERS];
    int recv_count;
    int recv[MAX_PEERS];
};

static int rank_on_node(int node, int local_rank) {
    return node * RANKS_PER_NODE + local_rank;
}

static struct peer_set peers_for(enum traffic_pattern pattern,
                                 int node,
                                 int local_rank,
                                 int rank) {
    struct peer_set result = {0};
    if (pattern == PATTERN_PAIRWISE_BIDIR) {
        result.send_count = 1;
        result.recv_count = 1;
        result.send[0] = rank_on_node(node ^ 1, local_rank);
        result.recv[0] = result.send[0];
    } else if (pattern == PATTERN_PD_2P2D_INCAST) {
        if (node == 0 || node == 2) {
            result.send_count = 2;
            result.send[0] = rank_on_node(1, local_rank);
            result.send[1] = rank_on_node(3, local_rank);
        } else {
            result.recv_count = 2;
            result.recv[0] = rank_on_node(0, local_rank);
            result.recv[1] = rank_on_node(2, local_rank);
        }
    } else if (pattern == PATTERN_PD_3P1D_INCAST) {
        if (node < 3) {
            result.send_count = 1;
            result.send[0] = rank_on_node(3, local_rank);
        } else {
            result.recv_count = 3;
            for (int source_node = 0; source_node < 3; ++source_node) {
                result.recv[source_node] = rank_on_node(source_node,
                                                        local_rank);
            }
        }
    } else {
        abort_with("traffic pattern has no peer mapping", rank);
    }
    return result;
}

static void exchange(unsigned char *send_buffer,
                     unsigned char *recv_buffer,
                     const struct config *config,
                     const struct peer_set *peers,
                     int node,
                     int tag_base,
                     MPI_Request *requests,
                     int rank) {
    const int count = (int)config->message_bytes;
    int request_index = 0;
    for (int peer_index = 0; peer_index < peers->recv_count; ++peer_index) {
        int peer = peers->recv[peer_index];
        int source_node = peer / RANKS_PER_NODE;
        for (int item = 0; item < config->inflight; ++item) {
            size_t offset = (
                (size_t)peer_index * (size_t)config->inflight
                + (size_t)item
            ) * config->message_bytes;
            int status = MPI_Irecv(
                recv_buffer + offset,
                count,
                MPI_BYTE,
                peer,
                tag_base + source_node * MAX_INFLIGHT + item,
                MPI_COMM_WORLD,
                &requests[request_index++]);
            if (status != MPI_SUCCESS) {
                abort_with("MPI_Irecv failed", rank);
            }
        }
    }
    for (int peer_index = 0; peer_index < peers->send_count; ++peer_index) {
        int peer = peers->send[peer_index];
        for (int item = 0; item < config->inflight; ++item) {
            unsigned char *source = (
                send_buffer + (size_t)item * config->message_bytes);
            int status = MPI_Isend(
                source,
                count,
                MPI_BYTE,
                peer,
                tag_base + node * MAX_INFLIGHT + item,
                MPI_COMM_WORLD,
                &requests[request_index++]);
            if (status != MPI_SUCCESS) {
                abort_with("MPI_Isend failed", rank);
            }
        }
    }
    int expected_requests = (
        peers->send_count + peers->recv_count) * config->inflight;
    if (request_index != expected_requests) {
        abort_with("request inventory mismatch", rank);
    }
    if (MPI_Waitall(request_index, requests, MPI_STATUSES_IGNORE)
        != MPI_SUCCESS) {
        abort_with("MPI_Waitall failed", rank);
    }
}

static int wait_for_start(const struct config *config, int rank) {
    if (config->start_file == NULL) {
        return 1;
    }
    const struct timespec poll_delay = {
        .tv_sec = 0,
        .tv_nsec = 10L * 1000L * 1000L,
    };
    for (;;) {
        int state = 0;
        if (rank == 0) {
            if (access(config->start_file, F_OK) == 0) {
                state = 1;
            } else if (errno != ENOENT) {
                abort_with("failed to inspect start file", rank);
            }
            if (state == 0 && config->stop_file != NULL) {
                if (access(config->stop_file, F_OK) == 0) {
                    state = 2;
                } else if (errno != ENOENT) {
                    abort_with("failed to inspect stop file before start", rank);
                }
            }
        }
        if (MPI_Bcast(&state, 1, MPI_INT, 0, MPI_COMM_WORLD)
            != MPI_SUCCESS) {
            abort_with("MPI_Bcast failed while waiting for start", rank);
        }
        if (state == 1) {
            return 1;
        }
        if (state == 2) {
            return 0;
        }
        struct timespec remaining = poll_delay;
        while (nanosleep(&remaining, &remaining) != 0 && errno == EINTR) {
        }
    }
}

int main(int argc, char **argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)
        != MPI_SUCCESS) {
        return 2;
    }
    int rank = -1;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (provided < MPI_THREAD_FUNNELED) {
        abort_with("MPI thread support is insufficient", rank);
    }
    if (world_size != EXPECTED_WORLD_SIZE) {
        abort_with("requires exactly 16 ranks (4 ranks on each of 4 nodes)", rank);
    }
    struct config config = parse_config(argc, argv, rank);
    int node = rank / RANKS_PER_NODE;
    int local_rank = rank % RANKS_PER_NODE;
    struct peer_set peers = peers_for(config.pattern, node, local_rank, rank);
    size_t batch_bytes = config.message_bytes * (size_t)config.inflight;
    size_t send_allocation_bytes = batch_bytes;
    size_t recv_allocation_bytes = batch_bytes * (
        peers.recv_count > 0 ? (size_t)peers.recv_count : 1U);
    unsigned char *send_buffer = NULL;
    unsigned char *recv_buffer = NULL;
    if (posix_memalign((void **)&send_buffer, 4096,
                       send_allocation_bytes) != 0
        || posix_memalign((void **)&recv_buffer, 4096,
                          recv_allocation_bytes) != 0) {
        abort_with("buffer allocation failed", rank);
    }
    memset(send_buffer, (unsigned char)(rank + 1), send_allocation_bytes);
    memset(recv_buffer, 0, recv_allocation_bytes);
    int request_count = (
        peers.send_count + peers.recv_count) * config.inflight;
    MPI_Request *requests = calloc((size_t)request_count, sizeof(*requests));
    if (requests == NULL) {
        abort_with("request allocation failed", rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int round = 0; round < WARMUP_ROUNDS; ++round) {
        exchange(send_buffer, recv_buffer, &config, &peers, node,
                 100 + round * TAG_BATCH_STRIDE, requests, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0 && config.ready_file != NULL) {
        FILE *ready = fopen(config.ready_file, "w");
        if (ready == NULL || fputs("ready\n", ready) == EOF
            || fclose(ready) != 0) {
            abort_with("failed to publish ready file", rank);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int start_observed = wait_for_start(&config, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    struct timespec started;
    struct timespec finished;
    clock_gettime(CLOCK_MONOTONIC, &started);
    finished = started;
    uint64_t iterations = 0;
    int stop_requested = start_observed ? 0 : 1;
    int keep_running = start_observed;
    while (keep_running) {
        struct timespec batch_begin;
        struct timespec batch_end;
        clock_gettime(CLOCK_MONOTONIC, &batch_begin);
        exchange(send_buffer, recv_buffer, &config, &peers, node,
                 1000 + (int)(iterations % 512U) * TAG_BATCH_STRIDE,
                 requests, rank);
        clock_gettime(CLOCK_MONOTONIC, &batch_end);
        ++iterations;
        throttle(seconds_between(&batch_begin, &batch_end), config.duty_cycle);
        clock_gettime(CLOCK_MONOTONIC, &finished);
        if (rank == 0 && config.stop_file != NULL) {
            if (access(config.stop_file, F_OK) == 0) {
                stop_requested = 1;
            } else if (errno != ENOENT) {
                abort_with("failed to inspect stop file", rank);
            }
        }
        int local_continue = seconds_between(&started, &finished) < config.duration_s;
        if (rank == 0 && stop_requested) {
            local_continue = 0;
        }
        MPI_Allreduce(&local_continue, &keep_running, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    clock_gettime(CLOCK_MONOTONIC, &finished);

    int local_correct = 1;
    for (int peer_index = 0; peer_index < peers.recv_count; ++peer_index) {
        const unsigned char expected = (
            (unsigned char)(peers.recv[peer_index] + 1));
        for (int item = 0; item < config.inflight; ++item) {
            size_t offset = (
                (size_t)peer_index * (size_t)config.inflight
                + (size_t)item
            ) * config.message_bytes;
            unsigned char *buffer = recv_buffer + offset;
            if (buffer[0] != expected
                || buffer[config.message_bytes / 2] != expected
                || buffer[config.message_bytes - 1] != expected) {
                local_correct = 0;
            }
        }
    }
    int all_correct = 0;
    MPI_Allreduce(&local_correct, &all_correct, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    double elapsed_s = seconds_between(&started, &finished);
    uint64_t sent_bytes = iterations * (uint64_t)config.inflight
                          * (uint64_t)config.message_bytes
                          * (uint64_t)peers.send_count;
    uint64_t received_bytes = iterations * (uint64_t)config.inflight
                              * (uint64_t)config.message_bytes
                              * (uint64_t)peers.recv_count;
    uint64_t aggregate_sent_bytes = 0;
    uint64_t aggregate_received_bytes = 0;
    uint64_t min_iterations = 0;
    uint64_t max_iterations = 0;
    double max_elapsed_s = 0.0;
    uint64_t rank_sent_bytes[EXPECTED_WORLD_SIZE] = {0};
    uint64_t rank_received_bytes[EXPECTED_WORLD_SIZE] = {0};
    MPI_Reduce(&sent_bytes, &aggregate_sent_bytes, 1, MPI_UINT64_T, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&received_bytes, &aggregate_received_bytes, 1, MPI_UINT64_T,
               MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Gather(&sent_bytes, 1, MPI_UINT64_T, rank_sent_bytes, 1,
               MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Gather(&received_bytes, 1, MPI_UINT64_T, rank_received_bytes, 1,
               MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Reduce(&iterations, &min_iterations, 1, MPI_UINT64_T, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&iterations, &max_iterations, 1, MPI_UINT64_T, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_s, &max_elapsed_s, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    if (rank == 0) {
        double aggregate_gbps = max_elapsed_s > 0.0
            ? ((double)aggregate_sent_bytes * 8.0) / max_elapsed_s / 1.0e9
            : 0.0;
        double aggregate_received_gbps = max_elapsed_s > 0.0
            ? ((double)aggregate_received_bytes * 8.0)
              / max_elapsed_s / 1.0e9
            : 0.0;
        uint64_t node_sent_bytes[NODE_COUNT] = {0};
        uint64_t node_received_bytes[NODE_COUNT] = {0};
        for (int item = 0; item < EXPECTED_WORLD_SIZE; ++item) {
            int item_node = item / RANKS_PER_NODE;
            node_sent_bytes[item_node] += rank_sent_bytes[item];
            node_received_bytes[item_node] += rank_received_bytes[item];
        }
        printf("{\"schema\":\"tempo-cxi-background-traffic-3\","
               "\"world_size\":%d,\"ranks_per_node\":%d,"
               "\"pattern\":\"%s\","
               "\"message_bytes\":%zu,\"inflight\":%d,"
               "\"duty_cycle\":%.6f,\"start_gated\":%s,"
               "\"start_observed\":%s,\"elapsed_s\":%.6f,"
               "\"min_iterations\":%llu,\"max_iterations\":%llu,"
               "\"aggregate_sent_bytes\":%llu,"
               "\"aggregate_received_bytes\":%llu,"
               "\"aggregate_sent_gbps\":%.6f,"
               "\"aggregate_received_gbps\":%.6f,"
               "\"node_sent_gbps\":[",
               world_size, RANKS_PER_NODE, pattern_name(config.pattern),
               config.message_bytes,
               config.inflight, config.duty_cycle,
               config.start_file != NULL ? "true" : "false",
               start_observed ? "true" : "false", max_elapsed_s,
               (unsigned long long)min_iterations,
               (unsigned long long)max_iterations,
               (unsigned long long)aggregate_sent_bytes,
               (unsigned long long)aggregate_received_bytes,
               aggregate_gbps, aggregate_received_gbps);
        for (int item_node = 0; item_node < NODE_COUNT; ++item_node) {
            double gbps = max_elapsed_s > 0.0
                ? ((double)node_sent_bytes[item_node] * 8.0)
                  / max_elapsed_s / 1.0e9
                : 0.0;
            printf("%s%.6f", item_node == 0 ? "" : ",", gbps);
        }
        printf("],\"node_received_gbps\":[");
        for (int item_node = 0; item_node < NODE_COUNT; ++item_node) {
            double gbps = max_elapsed_s > 0.0
                ? ((double)node_received_bytes[item_node] * 8.0)
                  / max_elapsed_s / 1.0e9
                : 0.0;
            printf("%s%.6f", item_node == 0 ? "" : ",", gbps);
        }
        printf("],\"stop_requested\":%s,\"correctness\":%s}\n",
               stop_requested ? "true" : "false",
               all_correct ? "true" : "false");
        fflush(stdout);
    }

    free(requests);
    free(recv_buffer);
    free(send_buffer);
    MPI_Finalize();
    return all_correct ? 0 : 3;
}
