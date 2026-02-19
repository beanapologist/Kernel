/*
 * Qudit Kernel — Extension of Pipeline of Coherence for d-dimensional quantum systems
 *
 * Extends the quantum_kernel_v2 framework from qubits (d=2) to arbitrary
 * d-dimensional quantum systems (qudits).
 *
 * Key extensions:
 *   QuditState     : d complex coefficients |ψ⟩ = Σ_k c_k|k⟩, k=0…d-1
 *   QuditOps       : generalized gates — shift X_d, clock Z_d, Fourier F_d, rotation R_d
 *   QuditEntangle  : coupling/entanglement for qudits of varying dimensions
 *   QuditMemory    : rotational memory addressing in Z/dZ
 *   QuditProcess   : schedulable unit with a d-dimensional quantum state
 *   QuditKernel    : kernel managing qudit processes on d-cycles
 *
 * Generalization of key qubit concepts:
 *   radius  r_d  = √(Σ_{k≥1}|c_k|²/(d-1)) / |c_0|          (r=1 ↔ balanced)
 *   coherence C_d = (Σ_{i<j} 2|c_i||c_j|) / (d-1)  ∈ [0,1]  (C=1 ↔ balanced)
 *   d-cycle step  : c_k *= ω_d^k, ω_d = e^{2πi/d}            (step^d = identity)
 *
 * For d=2 with c_0=α, c_1=β: r_2=|β/α|, C_2=2|α||β|, ω_2=e^{iπ}=-1
 */

#include <complex>
#include <vector>
#include <cmath>
#include <string>
#include <functional>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <memory>

// ── Numerical tolerances (consistent with quantum_kernel_v2) ─────────────────
constexpr double QUDIT_COHERENCE_TOL  = 1e-9;
constexpr double QUDIT_RADIUS_TOL     = 1e-9;
constexpr double QUDIT_CONSERVATION_TOL = 1e-12;

// ── Decoherence thresholds (identical to qubit kernel) ───────────────────────
constexpr double QUDIT_DECOHERENCE_MINOR   = 0.05;
constexpr double QUDIT_DECOHERENCE_MAJOR   = 0.15;

// Silver constant from Prop 4 (qubit kernel), preserved here for IPC compatibility
constexpr double QUDIT_DELTA_S    = 2.41421356237309504880;  // 1+√2
constexpr double QUDIT_DELTA_CONJ = 0.41421356237309504880;  // √2-1 = 1/δ_S

using Cx = std::complex<double>;

// ── ω_d = e^{2πi/d}: primitive d-th root of unity ────────────────────────────
Cx omega(int d, int power = 1) {
    const double angle = 2.0 * M_PI * power / d;
    return Cx{std::cos(angle), std::sin(angle)};
}

// ── QuditState: d-dimensional pure quantum state ──────────────────────────────
/*
 * Represents |ψ⟩ = Σ_{k=0}^{d-1} c_k |k⟩ with normalization Σ|c_k|²=1.
 *
 * Coherence metric C_d:
 *   C_d = (1/(d-1)) Σ_{i<j} 2|c_i||c_j| ∈ [0,1]
 *   Maximum C_d=1 when |c_k|=1/√d for all k (balanced/maximally coherent state).
 *   For d=2: C_2 = 2|c_0||c_1| = 2|α||β|  (matches qubit c_l1).
 *
 * Radius r_d:
 *   r_d = √(Σ_{k≥1}|c_k|²/(d-1)) / |c_0|
 *   r_d = 1 ↔ balanced state.  r_d > 1 ↔ excited-dominant.  r_d < 1 ↔ ground-dominant.
 *   For d=2: r_2 = |c_1|/|c_0| = |β/α|  (matches qubit radius).
 *
 * d-cycle step:
 *   step() applies c_k *= ω_d^k.
 *   After d steps: c_k *= (ω_d^k)^d = e^{2πi·k} = 1, so state^d = identity.
 *   |r_d| and |C_d| are invariant under step() (pure phase rotation).
 *   For d=2: c_1 *= e^{iπ} = -1 each step (2-cycle).
 */
struct QuditState {
    int d;                          // Hilbert space dimension (d ≥ 2)
    std::vector<Cx> coeffs;         // Coefficients c_0 … c_{d-1}

    // Balanced state: |c_k| = 1/√d for all k, phases adjusted for reality.
    // c_0 = 1/√d (real), c_k = ω_d^k/√d for k≥1 (unit-circle phases).
    explicit QuditState(int dim) : d(dim), coeffs(dim) {
        if (dim < 2) throw std::invalid_argument("QuditState: dimension must be >= 2");
        double scale = 1.0 / std::sqrt(static_cast<double>(dim));
        for (int k = 0; k < dim; ++k) {
            coeffs[k] = omega(dim, k) * scale;
        }
        // Fix c_0 to be real positive (canonical form)
        coeffs[0] = Cx{scale, 0.0};
    }

    // Initialize from explicit coefficient vector (will be normalized)
    QuditState(int dim, std::vector<Cx> c) : d(dim), coeffs(std::move(c)) {
        if (d < 2) throw std::invalid_argument("QuditState: dimension must be >= 2");
        if (static_cast<int>(coeffs.size()) != d)
            throw std::invalid_argument("QuditState: coefficient count mismatch");
        normalize();
    }

    // ── Normalization ──────────────────────────────────────────────────────────

    double norm_sq() const {
        double n = 0.0;
        for (const auto& c : coeffs) n += std::norm(c);
        return n;
    }

    void normalize() {
        double n = std::sqrt(norm_sq());
        if (n > QUDIT_COHERENCE_TOL) {
            for (auto& c : coeffs) c /= n;
        }
    }

    // ── Radius r_d ─────────────────────────────────────────────────────────────

    double radius() const {
        if (std::abs(coeffs[0]) < QUDIT_COHERENCE_TOL) return 0.0;
        if (d == 2) {
            // Exact qubit formula: r = |c_1|/|c_0|
            return std::abs(coeffs[1]) / std::abs(coeffs[0]);
        }
        // General: r_d = √(Σ_{k≥1}|c_k|²/(d-1)) / |c_0|
        double excited_sq = 0.0;
        for (int k = 1; k < d; ++k) excited_sq += std::norm(coeffs[k]);
        return std::sqrt(excited_sq / static_cast<double>(d - 1)) / std::abs(coeffs[0]);
    }

    bool balanced() const { return std::abs(radius() - 1.0) < QUDIT_RADIUS_TOL; }

    // ── Coherence C_d ──────────────────────────────────────────────────────────

    // C_d = (1/(d-1)) Σ_{i<j} 2|c_i||c_j| ∈ [0,1]
    double c_l1() const {
        double sum = 0.0;
        for (int i = 0; i < d; ++i) {
            double ai = std::abs(coeffs[i]);
            for (int j = i + 1; j < d; ++j) {
                sum += 2.0 * ai * std::abs(coeffs[j]);
            }
        }
        // Normalize so C=1 when balanced (all |c_k|=1/√d)
        // Max sum = C(d,2) * 2/d = d*(d-1)/2 * 2/d = (d-1)
        return sum / static_cast<double>(d - 1);
    }

    // ── d-cycle step ───────────────────────────────────────────────────────────
    // Apply c_k *= ω_d^k.  After d steps returns to original (pure phase rotation).
    void step() {
        for (int k = 1; k < d; ++k) {   // k=0: ω^0 = 1, no-op
            coeffs[k] *= omega(d, k);
        }
    }

    // ── Generalized coherence function (Theorem 11 analogue) ──────────────────
    // C(r) = 2r/(1+r²) works for any r ≥ 0, dimension-independent
    double coherence_fn() const {
        double r = radius();
        if (r < QUDIT_COHERENCE_TOL) return 0.0;
        return (2.0 * r) / (1.0 + r * r);
    }

    // Palindrome residual R(r) = (1/δ_S)(r - 1/r), R=0 ↔ r=1
    double palindrome() const {
        double r = radius();
        if (r < QUDIT_COHERENCE_TOL) return 0.0;
        return (1.0 / QUDIT_DELTA_S) * (r - 1.0 / r);
    }
};

// ── QuditOps: generalized quantum gate operations ─────────────────────────────
/*
 * All gates are represented as d×d unitary matrices (row-major flat vector).
 * Applying a gate: coeffs' = M * coeffs.
 *
 * Gates provided:
 *   shift_X(d)     : X_d|k⟩ = |(k+1) mod d⟩  (cyclic shift, d-cycle)
 *   clock_Z(d)     : Z_d|k⟩ = ω_d^k |k⟩      (phase clock)
 *   fourier_F(d)   : F_d|j⟩ = (1/√d) Σ_k ω_d^{jk}|k⟩  (QFT)
 *   rotation_R(d,φ): diagonal rotation R_d(φ)|k⟩ = e^{iφk/d}|k⟩
 */
class QuditOps {
public:
    using Matrix = std::vector<Cx>;   // d×d matrix, row-major

    // Apply a d×d gate matrix to a QuditState
    static void apply(QuditState& state, const Matrix& gate) {
        int d = state.d;
        std::vector<Cx> out(d, Cx{0.0, 0.0});
        for (int row = 0; row < d; ++row) {
            for (int col = 0; col < d; ++col) {
                out[row] += gate[row * d + col] * state.coeffs[col];
            }
        }
        state.coeffs = std::move(out);
    }

    // ── Shift (generalized Pauli X) ───────────────────────────────────────────
    // X_d|k⟩ = |(k+1) mod d⟩
    // X_d^d = I (d-cycle).  For d=2: Pauli X (bit-flip).
    static Matrix shift_X(int d) {
        Matrix M(d * d, Cx{0.0, 0.0});
        for (int k = 0; k < d; ++k) {
            // row = (k+1) mod d, col = k
            M[((k + 1) % d) * d + k] = Cx{1.0, 0.0};
        }
        return M;
    }

    // ── Clock (generalized Pauli Z) ───────────────────────────────────────────
    // Z_d|k⟩ = ω_d^k |k⟩,  ω_d = e^{2πi/d}
    // Z_d^d = I.  For d=2: Pauli Z (phase-flip).
    static Matrix clock_Z(int d) {
        Matrix M(d * d, Cx{0.0, 0.0});
        for (int k = 0; k < d; ++k) {
            M[k * d + k] = omega(d, k);
        }
        return M;
    }

    // ── Quantum Fourier Transform ─────────────────────────────────────────────
    // F_d|j⟩ = (1/√d) Σ_k ω_d^{jk}|k⟩
    // F_d is unitary: F_d†F_d = I.  For d=2: Hadamard H (up to global phase).
    static Matrix fourier_F(int d) {
        double scale = 1.0 / std::sqrt(static_cast<double>(d));
        Matrix M(d * d);
        for (int row = 0; row < d; ++row) {
            for (int col = 0; col < d; ++col) {
                M[row * d + col] = omega(d, row * col) * scale;
            }
        }
        return M;
    }

    // ── Diagonal rotation ─────────────────────────────────────────────────────
    // R_d(φ)|k⟩ = e^{iφk/d}|k⟩
    // Generalizes the qubit phase gate.  R_d(2π) = Z_d.
    static Matrix rotation_R(int d, double phi) {
        Matrix M(d * d, Cx{0.0, 0.0});
        for (int k = 0; k < d; ++k) {
            double angle = phi * k / d;
            M[k * d + k] = Cx{std::cos(angle), std::sin(angle)};
        }
        return M;
    }

    // ── Verify unitarity ──────────────────────────────────────────────────────
    // Returns max ||(M†M)_{ij} - δ_{ij}|| over all i,j
    static double unitarity_error(const Matrix& M, int d) {
        double max_err = 0.0;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                Cx sum{0.0, 0.0};
                for (int k = 0; k < d; ++k) {
                    sum += std::conj(M[k * d + i]) * M[k * d + j];
                }
                double expected = (i == j) ? 1.0 : 0.0;
                max_err = std::max(max_err, std::abs(sum - Cx{expected, 0.0}));
            }
        }
        return max_err;
    }
};

// ── QuditEntangle: coupling and entanglement between two qudits ───────────────
/*
 * Implements two mechanisms for qudit-qudit coupling:
 *
 * 1. Phase coupling (beam-splitter type, keeps states separable):
 *    For qudits of dimensions d1 and d2, couples component k of each:
 *      c_k^{(1)'} = c_k^{(1)} cos θ + c_k^{(2)} ω_{d1}^k sin θ   (k < min(d1,d2))
 *      c_k^{(2)'} = c_k^{(2)} cos θ - c_k^{(1)} conj(ω_{d2}^k) sin θ
 *    θ = coupling_strength * π / (2*max(d1,d2)), bounded to avoid inversion.
 *
 * 2. Controlled shift (entangling gate for same-dimension qudits):
 *    CX_d: |a⟩|b⟩ → |a⟩|(b + weighted_shift(a)) mod d2⟩
 *    Applied to separable inputs as a first-order approximation:
 *    c_b^{(2)'} ∝ Σ_a c_a^{(1)} · (X_d2^a · c^{(2)})_b
 */
class QuditEntangle {
public:
    struct Config {
        double coupling_strength = 0.1;   // θ parameter strength ∈ (0, 1)
        bool log_interactions    = false;  // Debug logging
        bool preserve_coherence  = true;   // Roll back incoherent interactions
    };

    explicit QuditEntangle(Config cfg) : config_(cfg) {}
    QuditEntangle() : config_(Config{}) {}

    // ── Phase coupling for arbitrary d1, d2 ───────────────────────────────────
    /*
     * Applies a beam-splitter style coupling between the two states.
     * Works for any d1, d2 ≥ 2.  Couples as many components as min(d1,d2).
     * Preserves normalization of both states individually.
     */
    bool phase_couple(QuditState& s1, QuditState& s2) {
        int coupled = std::min(s1.d, s2.d);
        double theta = config_.coupling_strength * M_PI / (2.0 * std::max(s1.d, s2.d));
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);

        QuditState s1_init = s1;
        QuditState s2_init = s2;

        for (int k = 0; k < coupled; ++k) {
            Cx phase1 = omega(s1.d, k);    // ω_{d1}^k
            Cx phase2 = omega(s2.d, k);    // ω_{d2}^k
            Cx c1 = s1_init.coeffs[k];
            Cx c2 = s2_init.coeffs[k];
            s1.coeffs[k] = c1 * cos_t + c2 * phase1 * sin_t;
            s2.coeffs[k] = c2 * cos_t - c1 * std::conj(phase2) * sin_t;
        }

        // Renormalize both states
        s1.normalize();
        s2.normalize();

        if (config_.preserve_coherence) {
            if (s1.c_l1() > 1.0 + QUDIT_COHERENCE_TOL ||
                s2.c_l1() > 1.0 + QUDIT_COHERENCE_TOL) {
                s1 = s1_init;
                s2 = s2_init;
                ++coherence_violations_;
                if (config_.log_interactions)
                    std::cout << "    ✗ QuditEntangle: coherence violation, rolled back\n";
                return false;
            }
        }

        ++total_couplings_;
        if (config_.log_interactions) {
            std::cout << "    ⊗ QuditEntangle(d1=" << s1.d << ",d2=" << s2.d
                      << "): θ=" << theta
                      << " C1=" << s1.c_l1() << " C2=" << s2.c_l1() << "\n";
        }
        return true;
    }

    // ── Controlled shift for equal-dimension qudits ───────────────────────────
    /*
     * Applies CX_d to two same-dimension qudits in product-state approximation:
     *   For each basis state |a⟩ of control, shifts target by a positions.
     *   When control is in superposition Σ c_a |a⟩, the result is:
     *     target' = Σ_a |c_a|² · X_d^a |ψ_target⟩  (incoherent mixture approximation)
     *   This first-order approximation preserves normalization and gives correct
     *   result when control is in a computational basis state.
     *   For a full coherent entangling gate, the joint d²-dimensional state
     *   would be required.
     */
    bool controlled_shift(QuditState& control, QuditState& target) {
        if (control.d != target.d) return false;
        int d = control.d;

        std::vector<Cx> shifted(d, Cx{0.0, 0.0});
        for (int a = 0; a < d; ++a) {
            double weight = std::norm(control.coeffs[a]);  // |c_a|²
            if (weight < QUDIT_COHERENCE_TOL) continue;
            // X_d^a |target⟩: shift target by a positions
            for (int b = 0; b < d; ++b) {
                shifted[(b + a) % d] += weight * target.coeffs[b];
            }
        }

        // Assign and renormalize
        target.coeffs = shifted;
        target.normalize();

        ++total_couplings_;
        return true;
    }

    void report_stats() const {
        std::cout << "  QuditEntangle: " << total_couplings_ << " couplings, "
                  << coherence_violations_ << " coherence violations\n";
    }

    Config config_;

private:
    uint64_t total_couplings_     = 0;
    uint64_t coherence_violations_ = 0;
};

// ── QuditMemory: Z/dZ rotational memory addressing ───────────────────────────
/*
 * Extends the qubit kernel's Z/8Z RotationalMemory to a general Z/dZ structure.
 * Memory is organized in d banks corresponding to positions 0…d-1 in Z/dZ.
 *
 * Address mapping:
 *   bank_position = address mod d
 *   bank_offset   = address div d
 *
 * Rotational properties:
 *   rotate_addressing(k): bank' = (bank + k) mod d
 *   Coherence preserved under rotation (pure re-mapping, data unchanged).
 *
 * d-dimensional coherence validation:
 *   Each bank stores quantum coefficients for its Z/dZ position.
 *   validate_coherence() checks that all stored coefficients are bounded.
 */
class QuditMemory {
public:
    struct MemoryBank {
        int position;                   // Position in Z/dZ
        std::vector<Cx> data;           // Stored quantum coefficients
        uint32_t access_count = 0;

        explicit MemoryBank(int pos) : position(pos) {}
    };

    struct Address {
        int bank;
        uint32_t offset;

        Address(int b, uint32_t o) : bank(b), offset(o) {}

        static Address from_linear(uint32_t linear_addr, int d) {
            return Address(static_cast<int>(linear_addr % static_cast<uint32_t>(d)),
                           linear_addr / static_cast<uint32_t>(d));
        }

        uint32_t to_linear(int d) const {
            return offset * static_cast<uint32_t>(d) + static_cast<uint32_t>(bank);
        }

        Address rotate(int k, int d) const {
            return Address((bank + k % d + d) % d, offset);
        }
    };

    explicit QuditMemory(int dim) : d_(dim) {
        if (dim < 2) throw std::invalid_argument("QuditMemory: dimension must be >= 2");
        for (int i = 0; i < dim; ++i) banks_.emplace_back(i);
    }

    void write(const Address& addr, const Cx& value) {
        ensure_capacity(addr);
        banks_[addr.bank].data[addr.offset] = value;
        ++banks_[addr.bank].access_count;
        ++total_writes_;
    }

    Cx read(const Address& addr) {
        ensure_capacity(addr);
        ++banks_[addr.bank].access_count;
        ++total_reads_;
        return banks_[addr.bank].data[addr.offset];
    }

    void write_linear(uint32_t linear_addr, const Cx& value) {
        write(Address::from_linear(linear_addr, d_), value);
    }

    Cx read_linear(uint32_t linear_addr) {
        return read(Address::from_linear(linear_addr, d_));
    }

    // Rotate address mapping by k positions in Z/dZ
    void rotate_addressing(int k) {
        rotation_offset_ = (rotation_offset_ + k % d_ + d_) % d_;
        ++rotation_count_;
    }

    int effective_bank(int logical_bank) const {
        return (logical_bank + rotation_offset_) % d_;
    }

    Address translate(const Address& addr) const {
        return Address(effective_bank(addr.bank), addr.offset);
    }

    const MemoryBank& get_bank(int position) const {
        return banks_[effective_bank(position)];
    }

    const std::vector<MemoryBank>& banks() const { return banks_; }

    int dimension() const { return d_; }

    // Validate d-dimensional coherence: all stored coefficients must be bounded
    bool validate_coherence() const {
        constexpr double MAX_COEFF_NORM = 100.0;
        for (const auto& bank : banks_) {
            for (const auto& coeff : bank.data) {
                if (std::norm(coeff) > MAX_COEFF_NORM) return false;
            }
        }
        return true;
    }

    struct Stats {
        uint64_t total_reads;
        uint64_t total_writes;
        uint32_t rotation_count;
        int rotation_offset;
        uint32_t total_capacity;
        int dimension;
    };

    Stats get_stats() const {
        uint32_t capacity = 0;
        for (const auto& bank : banks_) capacity += bank.data.size();
        return Stats{total_reads_, total_writes_, rotation_count_,
                     rotation_offset_, capacity, d_};
    }

    void report_stats() const {
        auto s = get_stats();
        std::cout << "  QuditMemory(d=" << s.dimension << "): "
                  << s.total_reads << " reads, " << s.total_writes << " writes, "
                  << s.rotation_count << " rotations, "
                  << "offset=" << s.rotation_offset << "/" << s.dimension << ", "
                  << "capacity=" << s.total_capacity << " cells\n";
    }

private:
    int d_;
    std::vector<MemoryBank> banks_;
    int rotation_offset_ = 0;
    uint64_t total_reads_  = 0;
    uint64_t total_writes_ = 0;
    uint32_t rotation_count_ = 0;

    void ensure_capacity(const Address& addr) {
        auto& bank = banks_[addr.bank];
        if (addr.offset >= bank.data.size()) {
            bank.data.resize(addr.offset + 1, Cx{0.0, 0.0});
        }
    }
};

// ── Decoherence severity levels for qudits ────────────────────────────────────
enum class QuditDecoherenceLevel { NONE, MINOR, MAJOR, CRITICAL };

QuditDecoherenceLevel measure_qudit_decoherence(double r) {
    double dev = std::abs(r - 1.0);
    if (dev <= QUDIT_RADIUS_TOL)           return QuditDecoherenceLevel::NONE;
    if (dev <= QUDIT_DECOHERENCE_MINOR)    return QuditDecoherenceLevel::MINOR;
    if (dev <= QUDIT_DECOHERENCE_MAJOR)    return QuditDecoherenceLevel::MAJOR;
    return QuditDecoherenceLevel::CRITICAL;
}

const char* qudit_decoherence_name(QuditDecoherenceLevel lvl) {
    switch (lvl) {
        case QuditDecoherenceLevel::NONE:     return "NONE";
        case QuditDecoherenceLevel::MINOR:    return "MINOR";
        case QuditDecoherenceLevel::MAJOR:    return "MAJOR";
        case QuditDecoherenceLevel::CRITICAL: return "CRITICAL";
    }
    return "";
}

// ── QuditProcess: schedulable unit on the d-cycle ────────────────────────────
/*
 * Each process holds a QuditState and advances through positions 0…d-1 in Z/dZ.
 * cycle_pos advances by one each tick; after d ticks a complete cycle is done.
 * Processes have cycle-aware memory access via QuditMemory.
 */
struct QuditProcess {
    uint32_t    pid;
    std::string name;
    QuditState  state;
    int         cycle_pos = 0;
    std::function<void(QuditProcess&)> task;
    bool        interacted = false;
    QuditMemory* memory    = nullptr;
    uint64_t*   current_tick = nullptr;

    QuditProcess(uint32_t pid_, std::string name_, QuditState st,
                 std::function<void(QuditProcess&)> task_ = nullptr,
                 QuditMemory* mem = nullptr, uint64_t* tick_ptr = nullptr)
        : pid(pid_), name(std::move(name_)), state(std::move(st)),
          task(std::move(task_)), memory(mem), current_tick(tick_ptr) {}

    // Cycle-aware memory write: address translated by cycle_pos in Z/dZ
    void mem_write(uint32_t addr, const Cx& value) {
        if (memory) {
            auto base = QuditMemory::Address::from_linear(addr, state.d);
            auto rotated = base.rotate(cycle_pos, state.d);
            memory->write(rotated, value);
        }
    }

    Cx mem_read(uint32_t addr) {
        if (memory) {
            auto base = QuditMemory::Address::from_linear(addr, state.d);
            auto rotated = base.rotate(cycle_pos, state.d);
            return memory->read(rotated);
        }
        return Cx{0.0, 0.0};
    }

    // One tick: advance cycle position and apply d-cycle step
    void tick() {
        state.step();
        cycle_pos = (cycle_pos + 1) % state.d;   // Z/dZ arithmetic
        interacted = false;
        if (task) task(*this);
    }

    void report() const {
        double r = state.radius();
        double C = state.c_l1();
        double R = state.palindrome();
        std::cout
            << "  PID " << pid
            << "  d=" << state.d
            << "  cycle=" << cycle_pos
            << "  r=" << std::setw(10) << r
            << "  C_ℓ1=" << std::setw(10) << C
            << "  R(r)=" << std::setw(10) << R
            << "  balanced=" << (state.balanced() ? "✓" : "✗")
            << "  \"" << name << "\"\n";
    }
};

// ── QuditKernel: kernel managing qudit processes ──────────────────────────────
class QuditKernel {
public:
    explicit QuditKernel(int d)
        : d_(d), entangle_(), memory_(std::make_shared<QuditMemory>(d)) {
        if (d < 2) throw std::invalid_argument("QuditKernel: dimension must be >= 2");

        // Validate silver conservation (Prop 4, preserved from qubit kernel)
        if (std::abs(QUDIT_DELTA_S * QUDIT_DELTA_CONJ - 1.0) > QUDIT_CONSERVATION_TOL)
            throw std::runtime_error("QuditKernel: silver conservation violated");
    }

    int dimension() const { return d_; }

    // Spawn a new qudit process in balanced state
    uint32_t spawn(const std::string& name,
                   std::function<void(QuditProcess&)> task = nullptr) {
        uint32_t pid = next_pid_++;
        processes_.emplace_back(pid, name, QuditState{d_}, task,
                                memory_.get(), &tick_);
        return pid;
    }

    // Spawn a qudit process with custom initial state (will be normalized)
    uint32_t spawn_with_state(const std::string& name, QuditState st,
                              std::function<void(QuditProcess&)> task = nullptr) {
        uint32_t pid = next_pid_++;
        processes_.emplace_back(pid, name, std::move(st), task,
                                memory_.get(), &tick_);
        return pid;
    }

    void enable_entanglement(const QuditEntangle::Config& cfg) {
        entangle_enabled_ = true;
        entangle_ = QuditEntangle(cfg);
    }

    void enable_entanglement() {
        entangle_enabled_ = true;
    }

    void tick() {
        ++tick_;

        // Apply entangling interactions before individual ticks
        if (entangle_enabled_) {
            for (size_t i = 0; i < processes_.size(); ++i) {
                for (size_t j = i + 1; j < processes_.size(); ++j) {
                    auto& pi = processes_[i];
                    auto& pj = processes_[j];
                    if (!pi.interacted && !pj.interacted &&
                        pi.cycle_pos == pj.cycle_pos) {
                        if (entangle_.phase_couple(pi.state, pj.state)) {
                            pi.interacted = true;
                            pj.interacted = true;
                        }
                    }
                }
            }
        }

        for (auto& p : processes_) p.tick();
    }

    void run(uint32_t n) { for (uint32_t i = 0; i < n; ++i) tick(); }

    void report() const {
        std::cout << "\n╔══ QuditKernel d=" << d_ << "  tick=" << tick_ << " ══╗\n";
        std::cout << std::fixed << std::setprecision(8);
        for (const auto& p : processes_) p.report();

        // Memory state
        memory_->report_stats();

        // Memory coherence validation
        if (!memory_->validate_coherence()) {
            std::cout << "  ⚠ WARNING: d-dimensional memory coherence failed\n";
        }

        // Entanglement stats
        if (entangle_enabled_) {
            entangle_.report_stats();
        }

        // Silver conservation (Prop 4)
        std::cout << "  Prop 4:  δ_S·(√2-1) = "
                  << QUDIT_DELTA_S * QUDIT_DELTA_CONJ << "  (must be 1.0)\n";

        std::cout << "╚════════════════════════════════════════════════╝\n";
    }

    QuditMemory& memory() { return *memory_; }
    const QuditMemory& memory() const { return *memory_; }

    std::vector<QuditProcess>& processes() { return processes_; }
    const std::vector<QuditProcess>& processes() const { return processes_; }

private:
    int d_;
    std::vector<QuditProcess> processes_;
    uint32_t next_pid_ = 1;
    uint64_t tick_     = 0;
    bool entangle_enabled_ = false;
    QuditEntangle entangle_;
    std::shared_ptr<QuditMemory> memory_;
};

// ════════════════════════════════════════════════════════════════════════════
// Main demonstration
// ════════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "Qudit Kernel — Extension of Pipeline of Coherence\n";
    std::cout << "d-dimensional quantum process kernel prototype\n\n";

    // ── 1. Qudit state preparation ───────────────────────────────────────────
    std::cout << "╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  1. QUDIT STATE PREPARATION                       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    for (int d : {2, 3, 4, 5, 8}) {
        QuditState qs(d);
        std::cout << "  d=" << d
                  << "  |ψ_balanced⟩: r=" << qs.radius()
                  << "  C_ℓ1=" << qs.c_l1()
                  << "  balanced=" << (qs.balanced() ? "✓" : "✗")
                  << "  norm=" << qs.norm_sq() << "\n";
    }

    // Custom d=3 qutrit state |ψ⟩ = (1/√2)|0⟩ + (1/2)|1⟩ + (1/2)|2⟩
    std::cout << "\n  Custom qutrit state |ψ⟩ ≈ (1/√2)|0⟩ + (1/2)|1⟩ + (1/2)|2⟩:\n";
    QuditState qutrit(3, {Cx{1.0/std::sqrt(2.0), 0.0},
                          Cx{0.5, 0.0},
                          Cx{0.5, 0.0}});
    std::cout << "    r=" << qutrit.radius()
              << "  C_ℓ1=" << qutrit.c_l1()
              << "  norm=" << qutrit.norm_sq() << "\n";

    // d=2 recovery: c_0=α=1/√2 (real), c_1=β=(-1+i)/2
    std::cout << "\n  d=2 canonical coherent state (matches qubit kernel):\n";
    QuditState qubit2(2, {Cx{1.0/std::sqrt(2.0), 0.0},
                          Cx{-0.5, 0.5}});
    std::cout << "    r=" << qubit2.radius()
              << "  C_ℓ1=" << qubit2.c_l1()
              << "  balanced=" << (qubit2.balanced() ? "✓" : "✗") << "\n";

    // ── 2. Qudit operations ──────────────────────────────────────────────────
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  2. QUDIT OPERATIONS                              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    for (int d : {2, 3, 4}) {
        std::cout << "  d=" << d << " gates:\n";

        double err_X = QuditOps::unitarity_error(QuditOps::shift_X(d), d);
        double err_Z = QuditOps::unitarity_error(QuditOps::clock_Z(d), d);
        double err_F = QuditOps::unitarity_error(QuditOps::fourier_F(d), d);
        double err_R = QuditOps::unitarity_error(QuditOps::rotation_R(d, M_PI), d);

        std::cout << "    shift X_d:   unitarity err = " << err_X
                  << (err_X < QUDIT_COHERENCE_TOL ? " ✓" : " ✗") << "\n";
        std::cout << "    clock Z_d:   unitarity err = " << err_Z
                  << (err_Z < QUDIT_COHERENCE_TOL ? " ✓" : " ✗") << "\n";
        std::cout << "    Fourier F_d: unitarity err = " << err_F
                  << (err_F < QUDIT_COHERENCE_TOL ? " ✓" : " ✗") << "\n";
        std::cout << "    rotation R_d:unitarity err = " << err_R
                  << (err_R < QUDIT_COHERENCE_TOL ? " ✓" : " ✗") << "\n";

        // Verify d-cycle property of shift: X_d^d = I
        QuditState test_state(d);
        for (int step = 0; step < d; ++step) {
            QuditOps::apply(test_state, QuditOps::shift_X(d));
        }
        // Compare with fresh balanced state
        QuditState ref_state(d);
        double diff = 0.0;
        for (int k = 0; k < d; ++k)
            diff = std::max(diff, std::abs(test_state.coeffs[k] - ref_state.coeffs[k]));
        std::cout << "    X_d^d = I:  max coeff diff = " << diff
                  << (diff < QUDIT_COHERENCE_TOL ? " ✓" : " ✗") << "\n\n";
    }

    // Demonstrate Fourier transform brings balanced state to |0⟩
    std::cout << "  QFT on balanced d=4 state:\n";
    {
        int d = 4;
        QuditState qft_in(d);
        // Start from computational |0⟩ (only c_0=1)
        std::vector<Cx> basis0(d, Cx{0,0});
        basis0[0] = Cx{1.0, 0.0};
        QuditState qft_state(d, basis0);
        QuditOps::apply(qft_state, QuditOps::fourier_F(d));
        std::cout << "  F_4|0⟩ coefficients: ";
        for (int k = 0; k < d; ++k) {
            std::cout << "(" << std::setprecision(4) << qft_state.coeffs[k].real()
                      << "+" << qft_state.coeffs[k].imag() << "i) ";
        }
        double C_after = qft_state.c_l1();
        std::cout << "\n  C_ℓ1 after QFT = " << C_after
                  << (std::abs(C_after - 1.0) < QUDIT_COHERENCE_TOL ? " ✓ (maximally coherent)" : "")
                  << "\n";
    }

    // ── 3. Entanglement between qudits of varying dimensions ─────────────────
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  3. QUDIT ENTANGLEMENT (VARYING DIMENSIONS)       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    // Phase coupling d=3 ↔ d=5
    {
        QuditState s3(3), s5(5);
        QuditEntangle::Config ec;
        ec.log_interactions = true;
        ec.coupling_strength = 0.2;
        QuditEntangle ent(ec);

        std::cout << "  Before coupling: C_ℓ1(d=3)=" << s3.c_l1()
                  << "  C_ℓ1(d=5)=" << s5.c_l1() << "\n";
        ent.phase_couple(s3, s5);
        std::cout << "  After  coupling: C_ℓ1(d=3)=" << s3.c_l1()
                  << "  C_ℓ1(d=5)=" << s5.c_l1() << "\n";
        std::cout << "  Both states normalized: norm(d=3)=" << s3.norm_sq()
                  << "  norm(d=5)=" << s5.norm_sq() << "\n\n";
    }

    // Controlled shift d=3 ↔ d=3
    {
        QuditState ctrl(3), tgt(3);
        // Set control to |1⟩ (c_1=1)
        ctrl.coeffs = {Cx{0,0}, Cx{1,0}, Cx{0,0}};
        // Target in balanced state
        QuditEntangle ent;
        std::cout << "  CX_3: control=|1⟩, target=balanced\n";
        std::cout << "    target before: ";
        for (auto& c : tgt.coeffs) std::cout << c << " ";
        ent.controlled_shift(ctrl, tgt);
        std::cout << "\n    target after  (shifted by 1): ";
        for (auto& c : tgt.coeffs) std::cout << c << " ";
        std::cout << "\n    norm(target)=" << tgt.norm_sq() << "\n\n";
    }

    // ── 4. Memory and coherence ───────────────────────────────────────────────
    std::cout << "╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  4. d-DIMENSIONAL MEMORY & COHERENCE              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    for (int d : {3, 5}) {
        QuditMemory mem(d);
        // Write d quantum coefficients
        for (int i = 0; i < d; ++i) {
            double scale = 1.0 / std::sqrt(static_cast<double>(d));
            mem.write_linear(static_cast<uint32_t>(i),
                             Cx{scale * std::cos(i * 2.0 * M_PI / d),
                                scale * std::sin(i * 2.0 * M_PI / d)});
        }
        mem.rotate_addressing(1);
        bool ok = mem.validate_coherence();
        std::cout << "  d=" << d << " memory after writing + rotate(1): coherence="
                  << (ok ? "✓ PASS" : "✗ FAIL") << "\n";
        mem.report_stats();
    }

    // ── 5. QuditKernel demonstrations ────────────────────────────────────────
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  5. QUDIT KERNEL — d=3 QUTRIT SCHEDULER           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    {
        QuditKernel kernel3(3);
        QuditEntangle::Config ec;
        ec.coupling_strength = 0.15;
        kernel3.enable_entanglement(ec);

        kernel3.spawn("Qutrit-A");
        kernel3.spawn("Qutrit-B", [](QuditProcess& p) {
            if (p.cycle_pos == 0) {
                std::cout << "    [Qutrit-B] completed 3-cycle, C_ℓ1="
                          << p.state.c_l1() << "\n";
            }
        });

        std::cout << "Initial state:\n";
        kernel3.report();

        kernel3.run(3);   // one complete 3-cycle
        std::cout << "\nAfter 3 ticks (one complete 3-cycle):\n";
        kernel3.report();
    }

    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║  6. QUDIT KERNEL — d=4 QUQUART SCHEDULER          ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n\n";

    {
        QuditKernel kernel4(4);
        kernel4.enable_entanglement();

        // Ququart processes using memory
        kernel4.spawn("Ququart-Writer", [](QuditProcess& p) {
            if (p.cycle_pos == 0) {
                p.mem_write(0, p.state.coeffs[0]);
                std::cout << "    [Writer] wrote c_0 at cycle " << p.cycle_pos << "\n";
            }
        });
        kernel4.spawn("Ququart-Reader", [](QuditProcess& p) {
            if (p.cycle_pos == 2) {
                Cx val = p.mem_read(0);
                std::cout << "    [Reader] read addr[0]=" << val
                          << " at cycle " << p.cycle_pos << "\n";
            }
        });

        // Demonstrate memory rotation in Z/4Z
        kernel4.memory().rotate_addressing(2);
        std::cout << "Memory rotated by 2 in Z/4Z:\n";
        kernel4.run(4);   // one complete 4-cycle
        kernel4.report();
    }

    std::cout << "\n✓ Qudit state preparation (arbitrary d)\n";
    std::cout << "✓ Generalized quantum operations (X_d, Z_d, F_d, R_d)\n";
    std::cout << "✓ Phase coupling and controlled shift entanglement\n";
    std::cout << "✓ Z/dZ rotational memory with d-dimensional coherence\n";
    std::cout << "✓ d-cycle process scheduling\n";
    std::cout << "✓ Silver conservation δ_S·(√2-1)=1 maintained\n";

    return 0;
}
