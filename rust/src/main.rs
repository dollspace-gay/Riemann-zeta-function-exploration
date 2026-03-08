use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;
use rayon::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const MAX_PRIME: usize = 1_000_000;
const DATA_DIR: &str = "~/rh_data";

fn expand_home(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn sieve_primes(max: usize) -> Vec<usize> {
    let sieve = primal::Sieve::new(max);
    sieve.primes_from(2).collect()
}

fn load_zeros(filename: &str) -> Vec<f64> {
    let path = expand_home(DATA_DIR).join(filename);
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<f64>().expect("Invalid number"))
        .collect()
}

fn build_gram_xi(
    n_zeros: usize,
    zeros: &[f64],
    primes: &[usize],
    log_primes: &[f64],
    sigma: f64,
    n_primes: usize,
    normalize: bool,
) -> DMatrix<f64> {
    let n = n_zeros.min(zeros.len());
    let np = n_primes.min(primes.len());
    let chunk_size = (np / rayon::current_num_threads().max(1)).max(100);

    let partial_matrices: Vec<DMatrix<f64>> = (0..np)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut g = DMatrix::<f64>::zeros(n, n);
            for &pi in chunk {
                let p = primes[pi] as f64;
                let logp = log_primes[pi];
                let w = logp * ((sigma - 0.5) * logp).cosh() / p.sqrt();
                let phases: Vec<Complex64> = (0..n)
                    .map(|i| Complex64::new(0.0, -zeros[i] * logp).exp())
                    .collect();
                for i in 0..n {
                    for j in i..n {
                        let val = w * (phases[i] * phases[j].conj()).re;
                        g[(i, j)] += val;
                        g[(j, i)] += val;
                    }
                }
            }
            g
        })
        .collect();

    let mut gram = DMatrix::<f64>::zeros(n, n);
    for g in partial_matrices {
        gram += g;
    }
    if normalize {
        let tr = gram.trace();
        if tr > 1e-15 { gram *= n as f64 / tr; }
    }
    gram
}

struct Stats { cond: f64, lambda_min: f64, lambda_max: f64 }

fn gram_stats(gram: &DMatrix<f64>) -> Stats {
    let eigen = SymmetricEigen::new(gram.clone());
    let eigvals = eigen.eigenvalues;
    let min = eigvals.min();
    let max = eigvals.max();
    Stats {
        cond: if min.abs() > 1e-15 { max / min } else { f64::INFINITY },
        lambda_min: min, lambda_max: max,
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RH CRYSTAL v3: Pushing to N=2000                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let total = Instant::now();

    let primes = sieve_primes(MAX_PRIME);
    let log_primes: Vec<f64> = primes.iter().map(|&p| (p as f64).ln()).collect();
    println!("Primes: {} (up to {})", primes.len(), MAX_PRIME);

    let zeta_zeros = load_zeros("zeta_zeros.csv");
    let max_n = zeta_zeros.len();
    println!("Zeta zeros: {}", max_n);
    println!("Max ratio at N=2000: {}", primes.len() / 2000);
    println!();

    // ==================================================================
    // TEST 1: THE BIG SCALING TEST
    // Ratio = 20, from N=100 to N=min(2000, max_zeros)
    // ==================================================================
    println!("{}", "=".repeat(66));
    println!("TEST 1: SCALING ratio=20 (N=100 to N=2000)");
    println!("{}", "=".repeat(66));

    let sizes_1: Vec<usize> = vec![
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        1200, 1400, 1600, 1800, 2000
    ];

    for &n in &sizes_1 {
        if n > max_n { break; }
        let np = (20 * n).min(primes.len());
        let t = Instant::now();
        let g = build_gram_xi(n, &zeta_zeros, &primes, &log_primes, 0.5, np, true);
        let s = gram_stats(&g);
        let elapsed = t.elapsed().as_secs_f64();
        println!(
            "  N={:5}, P={:6}: cond={:10.4}  lmin={:.8}  lmax={:.6}  ({:.2}s)",
            n, np, s.cond, s.lambda_min, s.lambda_max, elapsed
        );
    }

    // ==================================================================
    // TEST 2: SCALING ratio=39 (max sustainable ratio for N=2000)
    // ==================================================================
    println!("\n{}", "=".repeat(66));
    println!("TEST 2: SCALING ratio=39 (maximum for N=2000)");
    println!("{}", "=".repeat(66));

    for &n in &sizes_1 {
        if n > max_n { break; }
        let np = (39 * n).min(primes.len());
        let t = Instant::now();
        let g = build_gram_xi(n, &zeta_zeros, &primes, &log_primes, 0.5, np, true);
        let s = gram_stats(&g);
        let elapsed = t.elapsed().as_secs_f64();
        println!(
            "  N={:5}, P={:6}: cond={:10.4}  lmin={:.8}  lmax={:.6}  ({:.2}s)",
            n, np, s.cond, s.lambda_min, s.lambda_max, elapsed
        );
    }

    // ==================================================================
    // TEST 3: σ-DEPENDENCE AT N=500 WITH 25000 PRIMES
    // Does σ=1/2 remain the minimum at large N?
    // ==================================================================
    println!("\n{}", "=".repeat(66));
    println!("TEST 3: SIGMA DEPENDENCE (N=500, P=25000)");
    println!("{}", "=".repeat(66));

    let sigmas: Vec<f64> = (0..41).map(|i| 0.10 + 0.80 * (i as f64) / 40.0).collect();

    let results: Vec<(f64, f64, f64, f64)> = sigmas
        .par_iter()
        .map(|&sigma| {
            let g = build_gram_xi(500, &zeta_zeros, &primes, &log_primes, sigma, 25000, true);
            let s = gram_stats(&g);
            (sigma, s.cond, s.lambda_min, s.lambda_max)
        })
        .collect();

    let mut min_cond = f64::INFINITY;
    let mut min_sigma = 0.5;
    for &(sigma, cond, _, _) in &results {
        if cond < min_cond { min_cond = cond; min_sigma = sigma; }
    }
    println!("  Min condition: {:.6} at sigma={:.4}", min_cond, min_sigma);
    for &(sigma, cond, lmin, lmax) in &results {
        println!("    s={:.4}: cond={:.6}  lmin={:.8}  lmax={:.8}", sigma, cond, lmin, lmax);
    }

    // ==================================================================
    // TEST 4: σ-DEPENDENCE AT N=1000 (if we have enough zeros)
    // ==================================================================
    if max_n >= 1000 {
        println!("\n{}", "=".repeat(66));
        println!("TEST 4: SIGMA DEPENDENCE (N=1000, P=30000)");
        println!("{}", "=".repeat(66));

        let sigmas2: Vec<f64> = (0..21).map(|i| 0.20 + 0.60 * (i as f64) / 20.0).collect();

        let results2: Vec<(f64, f64, f64, f64)> = sigmas2
            .par_iter()
            .map(|&sigma| {
                let g = build_gram_xi(1000, &zeta_zeros, &primes, &log_primes, sigma, 30000, true);
                let s = gram_stats(&g);
                (sigma, s.cond, s.lambda_min, s.lambda_max)
            })
            .collect();

        let mut min_cond2 = f64::INFINITY;
        let mut min_sigma2 = 0.5;
        for &(sigma, cond, _, _) in &results2 {
            if cond < min_cond2 { min_cond2 = cond; min_sigma2 = sigma; }
        }
        println!("  Min condition: {:.6} at sigma={:.4}", min_cond2, min_sigma2);
        for &(sigma, cond, lmin, lmax) in &results2 {
            println!("    s={:.4}: cond={:.6}  lmin={:.8}  lmax={:.8}", sigma, cond, lmin, lmax);
        }
    }

    // ==================================================================
    // TEST 5: PRIME RATIO SWEEP AT N=500 AND N=1000
    // ==================================================================
    println!("\n{}", "=".repeat(66));
    println!("TEST 5: PRIME RATIO SWEEP");
    println!("{}", "=".repeat(66));

    let ratios = [5, 10, 15, 20, 30, 50, 75, 100, 150];

    println!("\n  N=500:");
    for &r in &ratios {
        let np = (r * 500).min(primes.len());
        let g = build_gram_xi(500, &zeta_zeros, &primes, &log_primes, 0.5, np, true);
        let s = gram_stats(&g);
        println!("    ratio={:4} ({:6} primes): cond={:8.4}  lmin={:.8}", r, np, s.cond, s.lambda_min);
    }

    if max_n >= 1000 {
        println!("\n  N=1000:");
        for &r in &ratios {
            let np = (r * 1000).min(primes.len());
            let g = build_gram_xi(1000, &zeta_zeros, &primes, &log_primes, 0.5, np, true);
            let s = gram_stats(&g);
            println!("    ratio={:4} ({:6} primes): cond={:8.4}  lmin={:.8}", r, np, s.cond, s.lambda_min);
        }
    }

    // ==================================================================
    // SUMMARY
    // ==================================================================
    println!("\n{}", "=".repeat(66));
    println!("DONE in {:.1}s", total.elapsed().as_secs_f64());
    println!("{}", "=".repeat(66));
}
