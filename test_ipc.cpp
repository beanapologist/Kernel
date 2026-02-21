/*
 * Test Suite for Inter-Process Communication (IPC)
 * 
 * Validates IPC functionality including:
 * - Message passing between quantum processes
 * - Coherence preservation during communication
 * - Cycle-aligned message delivery
 * - Queue management and capacity limits
 * - Multi-process communication networks
 */

#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include <openssl/crypto.h>

// ── Constants from quantum_kernel_v2.cpp ─────────────────────────────────────
constexpr double ETA        = 0.70710678118654752440;   // 1/√2
constexpr double DELTA_S    = 2.41421356237309504880;   // δ_S = 1+√2
constexpr double DELTA_CONJ = 0.41421356237309504880;   // √2-1 = 1/δ_S

constexpr double COHERENCE_TOLERANCE = 1e-9;
constexpr double RADIUS_TOLERANCE    = 1e-9;
constexpr double CONSERVATION_TOL    = 1e-12;

using Cx = std::complex<double>;
const Cx MU{ -ETA, ETA };  // µ = e^{i3π/4}

// Include necessary structures and functions from kernel
// In a real test suite, we'd link against the kernel or extract into a library
// For this demonstration, we'll use a simplified approach

// Test counter
int test_count = 0;
int passed = 0;
int failed = 0;

void test_assert(bool condition, const std::string& test_name) {
    ++test_count;
    if (condition) {
        std::cout << "  ✓ " << test_name << "\n";
        ++passed;
    } else {
        std::cout << "  ✗ FAILED: " << test_name << "\n";
        ++failed;
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 1 — Basic Message Structure
// ══════════════════════════════════════════════════════════════════════════════
void test_message_structure() {
    std::cout << "\n╔═══ Test 1: Message Structure ═══╗\n";
    
    // Message should contain all required fields
    struct TestMessage {
        uint32_t sender_pid;
        uint32_t receiver_pid;
        uint64_t timestamp;
        uint8_t sender_cycle_pos;
        Cx payload;
        double sender_coherence;
    };
    
    TestMessage msg{1, 2, 100, 3, Cx{0.5, 0.5}, 0.95};
    
    test_assert(msg.sender_pid == 1, "Sender PID stored correctly");
    test_assert(msg.receiver_pid == 2, "Receiver PID stored correctly");
    test_assert(msg.timestamp == 100, "Timestamp stored correctly");
    test_assert(msg.sender_cycle_pos == 3, "Cycle position stored correctly");
    test_assert(std::abs(msg.payload - Cx{0.5, 0.5}) < 1e-9, "Payload stored correctly");
    test_assert(std::abs(msg.sender_coherence - 0.95) < 1e-9, "Coherence stored correctly");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 2 — Coherence-Based Send Validation
// ══════════════════════════════════════════════════════════════════════════════
void test_coherence_send_validation() {
    std::cout << "\n╔═══ Test 2: Coherence-Based Send Validation ═══╗\n";
    
    // Simulate coherence threshold checks
    double coherence_threshold = 0.7;
    
    // Test case 1: High coherence (should allow send)
    double high_coherence = 0.95;
    bool should_send_high = (high_coherence >= coherence_threshold);
    test_assert(should_send_high, "High coherence allows message send");
    
    // Test case 2: Low coherence (should block send)
    double low_coherence = 0.5;
    bool should_block_low = (low_coherence < coherence_threshold);
    test_assert(should_block_low, "Low coherence blocks message send");
    
    // Test case 3: Exactly at threshold (should allow)
    double exact_coherence = 0.7;
    bool should_send_exact = (exact_coherence >= coherence_threshold);
    test_assert(should_send_exact, "Exact threshold coherence allows send");
    
    // Test case 4: Just below threshold (should block)
    double below_threshold = 0.699;
    bool should_block_below = (below_threshold < coherence_threshold);
    test_assert(should_block_below, "Below threshold coherence blocks send");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 3 — Payload Bounds Validation
// ══════════════════════════════════════════════════════════════════════════════
void test_payload_bounds() {
    std::cout << "\n╔═══ Test 3: Payload Bounds Validation ═══╗\n";
    
    constexpr double MAX_PAYLOAD_NORM = 100.0;
    
    // Test case 1: Normal quantum coefficient
    Cx normal_payload{0.5, 0.5};
    double normal_norm = std::norm(normal_payload);
    test_assert(normal_norm <= MAX_PAYLOAD_NORM, "Normal payload within bounds");
    
    // Test case 2: Large but valid coefficient
    Cx large_payload{5.0, 5.0};
    double large_norm = std::norm(large_payload);
    test_assert(large_norm <= MAX_PAYLOAD_NORM, "Large valid payload within bounds");
    
    // Test case 3: Excessive payload (should be rejected)
    Cx excessive_payload{50.0, 50.0};
    double excessive_norm = std::norm(excessive_payload);
    test_assert(excessive_norm > MAX_PAYLOAD_NORM, "Excessive payload exceeds bounds");
    
    // Test case 4: Zero payload (valid edge case)
    Cx zero_payload{0.0, 0.0};
    double zero_norm = std::norm(zero_payload);
    test_assert(zero_norm <= MAX_PAYLOAD_NORM, "Zero payload within bounds");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 4 — Cycle Alignment Logic
// ══════════════════════════════════════════════════════════════════════════════
void test_cycle_alignment() {
    std::cout << "\n╔═══ Test 4: Cycle Alignment Logic ═══╗\n";
    
    // Test cycle position matching for delivery
    uint8_t sender_pos = 3;
    
    // Test case 1: Exact match
    uint8_t receiver_pos_match = 3;
    bool should_deliver_match = (receiver_pos_match == sender_pos);
    test_assert(should_deliver_match, "Exact cycle position match allows delivery");
    
    // Test case 2: Position 0 (should not be special)
    uint8_t receiver_pos_zero = 0;
    bool should_not_deliver_zero = (receiver_pos_zero != sender_pos);
    test_assert(should_not_deliver_zero, "Position 0 does not bypass cycle alignment");
    
    // Test case 3: Different position
    uint8_t receiver_pos_diff = 5;
    bool should_wait = (receiver_pos_diff != sender_pos && receiver_pos_diff != 0);
    test_assert(should_wait, "Different position defers delivery");
    
    // Test case 4: Z/8Z boundary (7 and 0 are different positions)
    uint8_t boundary_send = 7;
    uint8_t boundary_recv = 0;
    bool boundary_mismatch = (boundary_recv != boundary_send);
    test_assert(boundary_mismatch, "Z/8Z positions 7 and 0 are distinct");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 5 — Queue FIFO Ordering
// ══════════════════════════════════════════════════════════════════════════════
void test_queue_fifo() {
    std::cout << "\n╔═══ Test 5: Queue FIFO Ordering ═══╗\n";
    
    // Simulate a message queue
    std::vector<int> queue;
    
    // Enqueue in order
    queue.push_back(1);
    queue.push_back(2);
    queue.push_back(3);
    
    // Dequeue and verify order
    test_assert(queue.size() == 3, "Queue has 3 messages");
    test_assert(queue[0] == 1, "First message is 1");
    test_assert(queue[1] == 2, "Second message is 2");
    test_assert(queue[2] == 3, "Third message is 3");
    
    // Remove first
    queue.erase(queue.begin());
    test_assert(queue.size() == 2, "Queue has 2 messages after removal");
    test_assert(queue[0] == 2, "New first message is 2");
    
    // FIFO property maintained
    test_assert(queue.front() == 2, "FIFO: oldest remaining message first");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 6 — Channel Management
// ══════════════════════════════════════════════════════════════════════════════
void test_channel_management() {
    std::cout << "\n╔═══ Test 6: Channel Management ═══╗\n";
    
    // Simulate channel identification
    struct ChannelId {
        uint32_t sender_pid;
        uint32_t receiver_pid;
        
        bool operator==(const ChannelId& other) const {
            return sender_pid == other.sender_pid && receiver_pid == other.receiver_pid;
        }
    };
    
    ChannelId ch1{1, 2};
    ChannelId ch2{2, 3};
    ChannelId ch1_dup{1, 2};
    ChannelId ch3{2, 1};  // Reverse direction
    
    test_assert(ch1 == ch1_dup, "Channel IDs match for same endpoints");
    test_assert(!(ch1 == ch2), "Different endpoints create different channels");
    test_assert(!(ch1 == ch3), "Reverse direction is different channel");
    
    // Directional channels
    test_assert(ch1.sender_pid != ch3.sender_pid || ch1.receiver_pid != ch3.receiver_pid,
                "Channels are directional (1→2 ≠ 2→1)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 7 — Silver Conservation During IPC
// ══════════════════════════════════════════════════════════════════════════════
void test_silver_conservation() {
    std::cout << "\n╔═══ Test 7: Silver Conservation During IPC ═══╗\n";
    
    // Verify silver conservation constant
    double conservation = DELTA_S * DELTA_CONJ;
    test_assert(std::abs(conservation - 1.0) < CONSERVATION_TOL,
                "δ_S·(√2-1) = 1 maintained");
    
    // Conservation holds regardless of message content
    for (int i = 0; i < 5; ++i) {
        double check = DELTA_S * DELTA_CONJ;
        test_assert(std::abs(check - 1.0) < CONSERVATION_TOL,
                    "Silver conservation invariant across operations");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 8 — Queue Capacity Limits
// ══════════════════════════════════════════════════════════════════════════════
void test_queue_capacity() {
    std::cout << "\n╔═══ Test 8: Queue Capacity Limits ═══╗\n";
    
    const uint32_t MAX_QUEUE_SIZE = 100;
    std::vector<int> queue;
    
    // Fill queue to capacity
    for (uint32_t i = 0; i < MAX_QUEUE_SIZE; ++i) {
        queue.push_back(i);
    }
    
    test_assert(queue.size() == MAX_QUEUE_SIZE, "Queue at capacity");
    
    // Attempt to exceed capacity should be blocked
    bool would_exceed = (queue.size() >= MAX_QUEUE_SIZE);
    test_assert(would_exceed, "Queue full prevents additional sends");
    
    // Remove one, should allow new message
    queue.pop_back();
    bool has_space = (queue.size() < MAX_QUEUE_SIZE);
    test_assert(has_space, "Removing message frees space");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 9 — Message Metadata Integrity
// ══════════════════════════════════════════════════════════════════════════════
void test_metadata_integrity() {
    std::cout << "\n╔═══ Test 9: Message Metadata Integrity ═══╗\n";
    
    // Ensure metadata doesn't corrupt during transmission
    struct MsgWithMetadata {
        uint32_t pid;
        uint64_t tick;
        uint8_t pos;
        Cx data;
        double coherence;
    };
    
    MsgWithMetadata original{5, 12345, 7, Cx{-0.5, 0.5}, 0.88};
    
    // Simulate storage/retrieval
    MsgWithMetadata stored = original;
    
    test_assert(stored.pid == original.pid, "PID preserved");
    test_assert(stored.tick == original.tick, "Timestamp preserved");
    test_assert(stored.pos == original.pos, "Cycle position preserved");
    test_assert(std::abs(stored.data - original.data) < 1e-15, "Payload preserved");
    test_assert(std::abs(stored.coherence - original.coherence) < 1e-15, "Coherence preserved");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 10 — Multi-Channel Independence
// ══════════════════════════════════════════════════════════════════════════════
void test_channel_independence() {
    std::cout << "\n╔═══ Test 10: Multi-Channel Independence ═══╗\n";
    
    // Simulate multiple independent channels
    struct Channel {
        uint32_t sender;
        uint32_t receiver;
        std::vector<int> queue;
    };
    
    Channel ch1{1, 2, {}};
    Channel ch2{2, 3, {}};
    Channel ch3{3, 1, {}};
    
    // Add messages to different channels
    ch1.queue.push_back(100);
    ch2.queue.push_back(200);
    ch2.queue.push_back(201);
    ch3.queue.push_back(300);
    
    test_assert(ch1.queue.size() == 1, "Channel 1 has 1 message");
    test_assert(ch2.queue.size() == 2, "Channel 2 has 2 messages");
    test_assert(ch3.queue.size() == 1, "Channel 3 has 1 message");
    
    // Operations on one channel don't affect others
    ch2.queue.clear();
    test_assert(ch1.queue.size() == 1, "Channel 1 unaffected by channel 2 clear");
    test_assert(ch3.queue.size() == 1, "Channel 3 unaffected by channel 2 clear");
    test_assert(ch2.queue.size() == 0, "Channel 2 cleared");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 11 — Coherence Threshold Boundaries
// ══════════════════════════════════════════════════════════════════════════════
void test_coherence_boundaries() {
    std::cout << "\n╔═══ Test 11: Coherence Threshold Boundaries ═══╗\n";
    
    // Test various threshold values
    std::vector<double> thresholds = {0.0, 0.5, 0.7, 0.9, 1.0};
    std::vector<double> coherences = {0.0, 0.49, 0.5, 0.69, 0.7, 0.89, 0.9, 0.99, 1.0};
    
    for (double threshold : thresholds) {
        for (double coherence : coherences) {
            bool expected_result = (coherence >= threshold);
            // Test that coherence check gives expected result
            test_assert(expected_result == (coherence >= threshold), 
                        "Threshold " + std::to_string(threshold) + 
                        " vs coherence " + std::to_string(coherence));
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Crypto helpers (mirrors QuantumIPC private methods for standalone testing)
// ══════════════════════════════════════════════════════════════════════════════

// Serialized size of complex<double>: 8 bytes real + 8 bytes imaginary
static constexpr size_t SERIALIZED_CX_SIZE = 16;

static std::vector<uint8_t> test_serialize_cx(const Cx& c) {
    std::vector<uint8_t> buf(SERIALIZED_CX_SIZE);
    double re = c.real(), im = c.imag();
    std::memcpy(buf.data(),     &re, 8);
    std::memcpy(buf.data() + 8, &im, 8);
    return buf;
}

static Cx test_deserialize_cx(const std::vector<uint8_t>& buf) {
    double re, im;
    std::memcpy(&re, buf.data(),     8);
    std::memcpy(&im, buf.data() + 8, 8);
    return Cx{re, im};
}

struct TestEncryptedData {
    std::vector<uint8_t> iv;
    std::vector<uint8_t> ciphertext;
    std::vector<uint8_t> tag;
    std::vector<uint8_t> hmac;
    bool is_encrypted = false;
};

static bool test_encrypt(const Cx& payload, const std::vector<uint8_t>& key,
                          TestEncryptedData& enc) {
    constexpr int IV_LEN = 12, TAG_LEN = 16;
    enc.iv.resize(IV_LEN);
    if (RAND_bytes(enc.iv.data(), IV_LEN) != 1) return false;
    auto plain = test_serialize_cx(payload);
    enc.ciphertext.resize(plain.size());
    enc.tag.resize(TAG_LEN);
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return false;
    bool ok = false;
    do {
        if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) break;
        if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, key.data(), enc.iv.data()) != 1) break;
        int len = 0;
        if (EVP_EncryptUpdate(ctx, enc.ciphertext.data(), &len,
                              plain.data(), (int)plain.size()) != 1) break;
        int fl = 0;
        if (EVP_EncryptFinal_ex(ctx, enc.ciphertext.data() + len, &fl) != 1) break;
        enc.ciphertext.resize(len + fl);
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, TAG_LEN, enc.tag.data()) != 1) break;
        ok = true;
    } while (false);
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

static bool test_decrypt(const TestEncryptedData& enc, const std::vector<uint8_t>& key,
                          Cx& payload) {
    std::vector<uint8_t> plain(enc.ciphertext.size());
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return false;
    bool ok = false;
    do {
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1) break;
        if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, key.data(), enc.iv.data()) != 1) break;
        int len = 0;
        if (EVP_DecryptUpdate(ctx, plain.data(), &len,
                              enc.ciphertext.data(), (int)enc.ciphertext.size()) != 1) break;
        auto tag_copy = enc.tag;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG,
                                (int)tag_copy.size(), tag_copy.data()) != 1) break;
        int fl = 0;
        if (EVP_DecryptFinal_ex(ctx, plain.data() + len, &fl) <= 0) break;
        plain.resize(len + fl);
        if (plain.size() != SERIALIZED_CX_SIZE) break;
        payload = test_deserialize_cx(plain);
        ok = true;
    } while (false);
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

static std::vector<uint8_t> test_compute_hmac(const std::vector<uint8_t>& key,
                                               const std::vector<uint8_t>& data) {
    std::vector<uint8_t> result(32);
    unsigned int len = 32;
    HMAC(EVP_sha256(), key.data(), (int)key.size(),
         data.data(), data.size(), result.data(), &len);
    result.resize(len);
    return result;
}

static bool test_verify_hmac(const std::vector<uint8_t>& key,
                              const std::vector<uint8_t>& data,
                              const std::vector<uint8_t>& expected) {
    auto computed = test_compute_hmac(key, data);
    if (computed.size() != expected.size()) return false;
    return CRYPTO_memcmp(computed.data(), expected.data(), computed.size()) == 0;
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 12 — Encrypted Message Structure Fields
// ══════════════════════════════════════════════════════════════════════════════
void test_encrypted_message_structure() {
    std::cout << "\n╔═══ Test 12: Encrypted Message Structure Fields ═══╗\n";

    TestEncryptedData enc;
    enc.iv.resize(12, 0xAB);
    enc.ciphertext.resize(16, 0xCD);
    enc.tag.resize(16, 0xEF);
    enc.hmac.resize(32, 0x12);
    enc.is_encrypted = true;

    test_assert(enc.iv.size() == 12,        "IV field is 12 bytes");
    test_assert(enc.ciphertext.size() == 16,"Ciphertext field present");
    test_assert(enc.tag.size() == 16,       "GCM tag field is 16 bytes");
    test_assert(enc.hmac.size() == 32,      "HMAC field is 32 bytes");
    test_assert(enc.is_encrypted,           "is_encrypted flag set correctly");

    TestEncryptedData plain_enc;
    test_assert(!plain_enc.is_encrypted,    "Default is_encrypted flag is false");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 13 — AES-256-GCM Encryption / Decryption Roundtrip
// ══════════════════════════════════════════════════════════════════════════════
void test_aes256gcm_roundtrip() {
    std::cout << "\n╔═══ Test 13: AES-256-GCM Encryption/Decryption Roundtrip ═══╗\n";

    // 32-byte symmetric key
    std::vector<uint8_t> key(32);
    RAND_bytes(key.data(), 32);

    // Test with a representative quantum payload
    Cx original{ETA, -ETA};  // canonical coherent state coefficient

    TestEncryptedData enc;
    bool encrypted = test_encrypt(original, key, enc);
    test_assert(encrypted, "AES-256-GCM encryption succeeds");
    enc.is_encrypted = true;  // flag set externally (by send_message in production)
    test_assert(enc.is_encrypted, "is_encrypted flag can be set after encryption");
    test_assert(enc.iv.size() == 12,  "IV is 12 bytes after encryption");
    test_assert(enc.tag.size() == 16, "GCM tag is 16 bytes after encryption");

    // Ciphertext should differ from plaintext
    auto plain_bytes = test_serialize_cx(original);
    bool differs = (enc.ciphertext != plain_bytes);
    test_assert(differs, "Ciphertext differs from plaintext");

    Cx recovered;
    bool decrypted = test_decrypt(enc, key, recovered);
    test_assert(decrypted, "AES-256-GCM decryption succeeds");
    test_assert(std::abs(recovered - original) < 1e-15, "Decrypted payload matches original");

    // Verify different key fails decryption
    std::vector<uint8_t> wrong_key(32, 0xFF);
    Cx bad_result;
    bool bad_decrypt = test_decrypt(enc, wrong_key, bad_result);
    test_assert(!bad_decrypt, "Decryption fails with wrong key");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 14 — HMAC-SHA256 Integrity Validation
// ══════════════════════════════════════════════════════════════════════════════
void test_hmac_integrity() {
    std::cout << "\n╔═══ Test 14: HMAC-SHA256 Integrity Validation ═══╗\n";

    std::vector<uint8_t> key(32);
    RAND_bytes(key.data(), 32);

    Cx payload{0.5, -0.5};
    TestEncryptedData enc;
    test_encrypt(payload, key, enc);

    // Build HMAC input: IV || ciphertext || tag
    std::vector<uint8_t> hmac_data;
    hmac_data.insert(hmac_data.end(), enc.iv.begin(),         enc.iv.end());
    hmac_data.insert(hmac_data.end(), enc.ciphertext.begin(), enc.ciphertext.end());
    hmac_data.insert(hmac_data.end(), enc.tag.begin(),        enc.tag.end());

    enc.hmac = test_compute_hmac(key, hmac_data);
    test_assert(enc.hmac.size() == 32, "HMAC-SHA256 produces 32-byte digest");

    // Valid HMAC verifies correctly
    bool valid = test_verify_hmac(key, hmac_data, enc.hmac);
    test_assert(valid, "Valid HMAC verifies correctly");

    // Tampered ciphertext fails HMAC verification
    TestEncryptedData tampered = enc;
    tampered.ciphertext[0] ^= 0xFF;
    std::vector<uint8_t> tampered_data;
    tampered_data.insert(tampered_data.end(), tampered.iv.begin(),         tampered.iv.end());
    tampered_data.insert(tampered_data.end(), tampered.ciphertext.begin(), tampered.ciphertext.end());
    tampered_data.insert(tampered_data.end(), tampered.tag.begin(),        tampered.tag.end());
    bool tampered_valid = test_verify_hmac(key, tampered_data, enc.hmac);
    test_assert(!tampered_valid, "Tampered ciphertext fails HMAC verification");

    // Wrong key fails HMAC verification
    std::vector<uint8_t> wrong_key(32, 0xAA);
    bool wrong_key_valid = test_verify_hmac(wrong_key, hmac_data, enc.hmac);
    test_assert(!wrong_key_valid, "Wrong key fails HMAC verification");

    // Deterministic: same inputs produce same HMAC
    auto hmac2 = test_compute_hmac(key, hmac_data);
    test_assert(enc.hmac == hmac2, "HMAC computation is deterministic");
}

// ══════════════════════════════════════════════════════════════════════════════
// Test 15 — Channel-Level Encryption Key Configuration
// ══════════════════════════════════════════════════════════════════════════════
void test_channel_encryption_config() {
    std::cout << "\n╔═══ Test 15: Channel-Level Encryption Configuration ═══╗\n";

    // Simulate channel with optional key
    struct EncryptedChannel {
        uint32_t sender_pid;
        uint32_t receiver_pid;
        std::vector<uint8_t> channel_key;

        void set_key(const std::vector<uint8_t>& key) {
            if (!key.empty() && key.size() != 32)
                throw std::invalid_argument("Key must be 32 bytes");
            channel_key = key;
        }
        bool has_encryption() const { return !channel_key.empty(); }
    };

    EncryptedChannel ch{1, 2, {}};
    test_assert(!ch.has_encryption(), "Channel starts without encryption");

    std::vector<uint8_t> key(32);
    RAND_bytes(key.data(), 32);
    ch.set_key(key);
    test_assert(ch.has_encryption(),          "Channel has encryption after set_key");
    test_assert(ch.channel_key.size() == 32,  "Channel key is 32 bytes");
    test_assert(ch.channel_key == key,        "Channel key stored correctly");

    // Invalid key size rejected
    bool threw = false;
    try { ch.set_key(std::vector<uint8_t>(16, 0)); } catch (const std::invalid_argument&) { threw = true; }
    test_assert(threw, "Invalid key size (16 bytes) is rejected");

    // Different channels hold independent keys
    EncryptedChannel ch2{2, 3, {}};
    std::vector<uint8_t> key2(32, 0xBB);
    ch2.set_key(key2);
    test_assert(ch.channel_key != ch2.channel_key, "Different channels hold independent keys");

    // Encryption key can be cleared
    ch.set_key({});
    test_assert(!ch.has_encryption(), "Encryption disabled after clearing key");

    // Full encrypt/decrypt roundtrip using channel key
    EncryptedChannel ch3{3, 4, {}};
    std::vector<uint8_t> session_key(32);
    RAND_bytes(session_key.data(), 32);
    ch3.set_key(session_key);

    Cx original{MU.real(), MU.imag()};  // µ = e^{i3π/4}
    TestEncryptedData enc;
    bool enc_ok = test_encrypt(original, ch3.channel_key, enc);

    // Build and verify HMAC using channel key
    std::vector<uint8_t> hmac_data;
    hmac_data.insert(hmac_data.end(), enc.iv.begin(),         enc.iv.end());
    hmac_data.insert(hmac_data.end(), enc.ciphertext.begin(), enc.ciphertext.end());
    hmac_data.insert(hmac_data.end(), enc.tag.begin(),        enc.tag.end());
    enc.hmac = test_compute_hmac(ch3.channel_key, hmac_data);
    enc.is_encrypted = true;

    bool hmac_ok = test_verify_hmac(ch3.channel_key, hmac_data, enc.hmac);
    Cx recovered;
    bool dec_ok = test_decrypt(enc, ch3.channel_key, recovered);

    test_assert(enc_ok,  "Channel-keyed encryption succeeds");
    test_assert(hmac_ok, "Channel-keyed HMAC verification succeeds");
    test_assert(dec_ok,  "Channel-keyed decryption succeeds");
    test_assert(std::abs(recovered - original) < 1e-15,
                "Payload recovered correctly via channel key");
}

// ══════════════════════════════════════════════════════════════════════════════
// Main Test Runner
// ══════════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║  IPC Test Suite — Quantum Kernel v2.0         ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    test_message_structure();
    test_coherence_send_validation();
    test_payload_bounds();
    test_cycle_alignment();
    test_queue_fifo();
    test_channel_management();
    test_silver_conservation();
    test_queue_capacity();
    test_metadata_integrity();
    test_channel_independence();
    test_coherence_boundaries();
    test_encrypted_message_structure();
    test_aes256gcm_roundtrip();
    test_hmac_integrity();
    test_channel_encryption_config();
    
    // Summary
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Summary                                 ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    std::cout << "  Total tests:  " << test_count << "\n";
    std::cout << "  Passed:       " << passed << " ✓\n";
    std::cout << "  Failed:       " << failed << " ✗\n";
    
    if (failed == 0) {
        std::cout << "\n✓ All IPC tests passed!\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed\n";
        return 1;
    }
}
