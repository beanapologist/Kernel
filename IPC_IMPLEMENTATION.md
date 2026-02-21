# Inter-Process Communication (IPC) Implementation

## Overview

This document describes the IPC functionality added to the Quantum Kernel v2.0, which enables coherence-preserving message passing between quantum processes.

## Architecture

### Core Components

1. **QuantumIPC Class** (`quantum_kernel_v2.cpp`, lines 602-894)
   - Manages message queues and channels
   - Enforces coherence validation
   - Handles cycle alignment

2. **Message Structure**
   - `sender_pid`: Source process ID
   - `receiver_pid`: Destination process ID
   - `timestamp`: Tick when sent
   - `sender_cycle_pos`: Z/8Z position at send time
   - `payload`: Complex quantum coefficient
   - `sender_coherence`: C(r) value for validation
   - `encrypted`: AES-256-GCM encrypted payload data (IV, ciphertext, GCM tag, HMAC-SHA256)

3. **Process Integration**
   - `send_to(pid, payload)`: Send message
   - `receive_from(pid)`: Receive messages
   - `pending_from(pid)`: Check queue depth

### Key Features

#### 1. Coherence Preservation

Messages are gated by coherence thresholds:
- **Send validation**: Sender must have C(r) ≥ threshold
- **Receive validation**: Receiver must have C(r) ≥ threshold
- **Default threshold**: 0.5 (configurable)

This prevents decoherent processes from spreading incoherence through communication.

#### 2. Z/8Z Cycle Alignment

Messages respect the 8-cycle scheduler:
- Messages sent at position k can only be received at position k
- Strict enforcement prevents violation of rotational invariants
- Maintains temporal coherence across the cycle

#### 3. Queue Management

- **FIFO ordering**: First-in-first-out delivery
- **Capacity limits**: Default max 100 messages per channel
- **Multi-channel**: Independent queues for each sender-receiver pair
- **Directional**: Channel from A→B is distinct from B→A

#### 4. Payload Validation

- Quantum coefficients must satisfy ||payload||² ≤ 100.0
- Prevents numerical overflow
- Maintains bounded state space

#### 5. Classical Cryptographic Security (AES-256-GCM + HMAC-SHA256)

When a channel has a symmetric key configured (via `set_channel_key`):
- **Encryption**: Payload is encrypted with AES-256-GCM (256-bit key, random 12-byte IV) before queuing
- **Integrity**: HMAC-SHA256 is computed over IV ∥ ciphertext ∥ GCM tag and stored with the message
- **Delivery**: HMAC is verified (constant-time) and payload is decrypted before delivery; messages failing integrity checks are silently dropped
- **Per-channel keys**: Each directional channel has its own independent symmetric key

Channels without a key operate identically to prior behavior (backward-compatible).

## Configuration

```cpp
QuantumIPC::Config cfg;
cfg.enable_coherence_check = true;      // Gate by C(r)
cfg.enable_cycle_alignment = true;      // Enforce Z/8Z matching
cfg.coherence_threshold = 0.7;          // Minimum C(r)
cfg.max_queue_size = 100;               // Per-channel limit
cfg.log_messages = true;                // Debug logging

kernel.enable_ipc(cfg);

// Enable AES-256-GCM encryption on a specific channel
std::vector<uint8_t> key(32);           // 32-byte symmetric key
RAND_bytes(key.data(), 32);
kernel.ipc().set_channel_key(1, 2, key); // Encrypt channel PID 1 → PID 2
```

## Usage Examples

### Basic Send/Receive

```cpp
kernel.spawn("Sender", [](Process& p) {
    if (p.cycle_pos == 3) {
        p.send_to(2, p.state.beta);  // Send to PID 2
    }
});

kernel.spawn("Receiver", [](Process& p) {
    if (p.cycle_pos == 3) {
        auto msgs = p.receive_from(1);  // From PID 1
        for (const auto& msg : msgs) {
            // Process message payload
        }
    }
});
```

### Communication Ring

```cpp
// Node-1 → Node-2 → Node-3 → Node-1
kernel.spawn("Node-1", [](Process& p) {
    if (p.cycle_pos == 0) p.send_to(2, p.state.beta);
    if (p.cycle_pos == 4) p.receive_from(3);
});

kernel.spawn("Node-2", [](Process& p) {
    if (p.cycle_pos == 0) p.receive_from(1);
    if (p.cycle_pos == 2) p.send_to(3, p.state.beta);
});

kernel.spawn("Node-3", [](Process& p) {
    if (p.cycle_pos == 2) p.receive_from(2);
    if (p.cycle_pos == 4) p.send_to(1, p.state.beta);
});
```

## Mathematical Properties

### Coherence Preservation

For a message to be transmitted:
1. C_sender(r) ≥ threshold
2. C_receiver(r) ≥ threshold
3. ||payload||² ≤ MAX_PAYLOAD_NORM

This ensures that communication only occurs between sufficiently coherent processes, preventing decoherence propagation.

### Cycle Alignment

If message sent at position k ∈ Z/8Z, delivery only at position k:
- Preserves rotational symmetry
- Maintains phase coherence
- Respects cycle boundaries

### Silver Conservation

IPC operations do not modify process states directly, only exchange information. The fundamental invariant δ_S·(√2-1) = 1 is preserved throughout all IPC operations.

## Testing

### Test Coverage

1. **Unit Tests** (`test_ipc.cpp`): 124 tests covering:
   - Message structure integrity
   - Coherence validation logic
   - Payload bounds checking
   - Cycle alignment enforcement
   - Queue FIFO ordering
   - Channel independence
   - Metadata preservation
   - AES-256-GCM encryption/decryption roundtrip
   - HMAC-SHA256 integrity and tamper detection
   - Channel-level encryption key configuration

2. **Integration Tests**: Kernel demonstrations showing:
   - Basic message passing
   - Coherence-gated blocking
   - Multi-process networks
   - Cycle-aligned delivery

### Test Results

```
Total tests:  124
Passed:       124 ✓
Failed:       0 ✗
```

No regression in existing theorem tests (56 tests) or NIST interrupt tests.

## Performance Characteristics

- **Send**: O(1) - constant time queue insertion
- **Receive**: O(m) - linear in pending messages for that channel
- **Memory**: O(n·k) - n processes, k average queue depth
- **Coherence check**: O(1) - single C(r) evaluation

## Security Considerations

1. **Coherence Gating**: Prevents decoherent processes from communicating
2. **Bounded Queues**: Prevents resource exhaustion
3. **Payload Limits**: Prevents numerical overflow attacks
4. **Process Isolation**: Each channel is independent
5. **No State Corruption**: IPC doesn't modify sender/receiver states
6. **AES-256-GCM Encryption**: Optional per-channel payload encryption with authenticated encryption
7. **HMAC-SHA256 Integrity**: Message integrity verified before decryption; constant-time comparison prevents timing attacks
8. **Per-Channel Keys**: Independent symmetric keys per directional channel limit blast radius of key compromise

## Future Enhancements

Potential improvements (not currently implemented):
- Priority-based message delivery
- Broadcast/multicast support
- Adaptive coherence thresholds
- Message compression for large payloads
- Key rotation and ephemeral session keys

## References

- Quantum Kernel v2.0 source code
- Pipeline of Coherence mathematical theorems
- Test suite documentation
- NIST evaluation standards

---

**Implementation Date**: 2026-02-19  
**Version**: 1.0  
**Author**: GitHub Copilot Agent
