#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <math.h>       /* exp */
#include <stdint.h>
#include <time.h>


double inline partial_gradient(double p, double b) {
    // partial gradient of logistic loss
    double phi, exp_t;
    p = p * b;
    if (p > 0)
        phi = 1. / (1. + exp(-p));
    else {
        exp_t = exp(p);
        phi = exp_t / (1. + exp_t);
    }
    return (phi - 1) * b;
}

/* L1 proximal operator */
double inline prox(double x, double step_size) {
    return fmax(x - step_size, 0) - fmax(- x - step_size, 0);
}

/* set foo = foo + bar atomically */
void inline add_atomic(std::atomic<double>* foo, double bar) {
  auto current = foo[0].load();
  while (!foo[0].compare_exchange_weak(current, current + bar))
    ;
}

void saga_single_thread(
        std::atomic<double>* x, std::atomic<double>* memory_gradient, std::atomic<double>* gradient_average,
        double* A_data, int64_t* A_indices, int64_t* A_indptr, double* b,
        double* d, int64_t n_samples, int64_t n_features, double alpha,
        double beta, double step_size, int64_t max_iter, double* trace_x,
        double* trace_time, int thread_id, int64_t iter_freq) {
    int64_t i, j, j_idx, local_counter=0, global_counter;
    double p, grad_i, incr, old_grad, delta;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int64_t> uni(0, n_samples-1);
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (true) {
      /* take a snapshot of the current vector of iterates in trace_x
         and time in trace_time, in order to plot convergence later on */
      if (local_counter % iter_freq == 0 && thread_id == 0) {
          int64_t c = local_counter / iter_freq;
          for (j=0; j<n_features; j++) {
              trace_x[n_features * c + j] = x[j];
          }
          clock_gettime(CLOCK_MONOTONIC, &finish);
          elapsed = (finish.tv_sec - start.tv_sec);
          elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
          trace_time[c] = (double) elapsed;
          printf(".. iteration %lld, time elapsed %f min ..\n", c, elapsed / 60.);
      }

      i = uni(rng);
      p = 0.;
      for (j=A_indptr[i]; j < A_indptr[i+1]; j++) {
          j_idx = A_indices[j];
          p += x[j_idx] * A_data[j];
      }
      grad_i = partial_gradient(p, b[i]);
      old_grad = memory_gradient[i].load();
      while (!memory_gradient[i].compare_exchange_weak(old_grad, grad_i))
          ;
      incr = grad_i - old_grad;

      // .. update coefficients ..
      for (j=A_indptr[i]; j < A_indptr[i+1]; j++){
          j_idx = A_indices[j];
          delta = incr * A_data[j] + d[j_idx] * (gradient_average[j_idx] + alpha * x[j_idx]);
          x[j_idx] = prox(x[j_idx] - step_size * delta, beta * step_size * d[j_idx]);
          add_atomic(&gradient_average[j_idx], incr * A_data[j] / n_samples);
      }
      local_counter ++;
      if (local_counter >= iter_freq * max_iter){
          if (thread_id == 0) {
              printf("..  done %lld iterations ..\n", local_counter / iter_freq);
          }
          return;
      }
  }

}


void saga_single_thread_nonatomic(
        double* x, double* memory_gradient, double* gradient_average,
        double* A_data, int64_t* A_indices, int64_t* A_indptr, double* b, double* d, int64_t n_samples,
        int64_t n_features, double alpha, double beta, double step_size, int64_t max_iter,
        double* trace_x, double* trace_time, int64_t iter_freq) {
          int64_t i, j, j_idx, local_counter=0, global_counter;
          double p, grad_i, incr, old_grad, delta;
          std::random_device rd;
          std::mt19937 rng(rd());
          std::uniform_int_distribution<int64_t> uni(0, n_samples-1);
          struct timespec start, finish;
          double elapsed;
          clock_gettime(CLOCK_MONOTONIC, &start);

          while (true) {
            /* take a snapshot of the current vector of iterates in trace_x
               and time in trace_time, in order to plot convergence later on */
            if (local_counter % iter_freq == 0) {
                int64_t c = local_counter / iter_freq;
                for (j=0; j<n_features; j++) {
                    trace_x[n_features * c + j] = x[j];
                }
                clock_gettime(CLOCK_MONOTONIC, &finish);
                elapsed = (finish.tv_sec - start.tv_sec);
                elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
                trace_time[c] = (double) elapsed;
                printf(".. iteration %lld, time elapsed %f min ..\n", c, elapsed / 60.);
            }

            i = uni(rng);
            p = 0.;
            for (j=A_indptr[i]; j < A_indptr[i+1]; j++) {
                j_idx = A_indices[j];
                p += x[j_idx] * A_data[j];
            }
            grad_i = partial_gradient(p, b[i]);
            old_grad = memory_gradient[i];
            memory_gradient[i] = grad_i;
            incr = grad_i - old_grad;

            // .. update coefficients ..
            for (j=A_indptr[i]; j < A_indptr[i+1]; j++){
                j_idx = A_indices[j];
                delta = incr * A_data[j] + d[j_idx] * (gradient_average[j_idx] + alpha * x[j_idx]);
                x[j_idx] = prox(x[j_idx] - step_size * delta, beta * step_size * d[j_idx]);
                gradient_average[j_idx] += incr * A_data[j] / n_samples;
            }
            local_counter ++;
            if (local_counter >= iter_freq * max_iter){
                printf("..  done %lld iterations ..\n", local_counter / iter_freq);
                return;
            }
        }
}


int prox_asaga(
        double* x, double* A_data, int64_t* A_indices, int64_t* A_indptr,
        double* b, double* d, int64_t n_samples, int64_t n_features,
        int64_t n_threads, double alpha, double beta, double step_size,
        int64_t max_iter, double* trace_x, double* trace_time, int64_t iter_freq) {

    std::vector<std::thread> threads;
    std::atomic<double>* memory_gradient;
    std::atomic<double>* gradient_average;
    double* memory_gradient_nonatomic;
    double* gradient_average_nonatomic;

    if (n_threads == 1) {
      memory_gradient_nonatomic = new double[n_samples];
      gradient_average_nonatomic = new double[n_features];
      for(int i=0; i<n_samples; i++){
        memory_gradient_nonatomic[i] = 0;
      }
      for(int j=0; j<n_features; j++){
        gradient_average_nonatomic[j] = 0;
      }
      saga_single_thread_nonatomic(x, memory_gradient_nonatomic,
        gradient_average_nonatomic, A_data, A_indices, A_indptr,
        b, d, n_samples, n_features, alpha, beta, step_size, max_iter,
        trace_x, trace_time, iter_freq);
    } else {
      memory_gradient = new std::atomic<double>[n_samples];
      for(int i=0; i<n_samples; i++){
        memory_gradient[i] = 0;
      }
      gradient_average = new std::atomic<double>[n_features];
      for(int j=0; j<n_features; j++){
        gradient_average[j] = 0;
      }

      std::atomic<double>* x_atomic = new std::atomic<double>[n_features];
      for(int j=0; j<n_features; j++){
        x_atomic[j] = x[j];
      }

      for (int i = 0; i < n_threads; ++i) {
          threads.push_back(std::thread(saga_single_thread, x_atomic, memory_gradient, gradient_average, A_data,
              A_indices, A_indptr, b, d, n_samples, n_features, alpha, beta, step_size, max_iter,
              trace_x, trace_time, i, iter_freq));
      }

      for(auto &t : threads){
          t.join();
      }
      for(int j=0; j<n_features; j++){
        x[j] = x_atomic[j];
      }
      delete[] memory_gradient;
      delete[] gradient_average;

    }

    return 0;
}


/* expose the above prox_asaga function so that it can be called from */
extern "C"
{
    extern int cffi_prox_asaga(double* x, double* A_data, int64_t* A_indices, int64_t* A_indptr, double* b,
        double* d, int64_t n_samples, int64_t n_features, int64_t n_threads, double alpha, double beta,
        double step_size, int64_t max_iter, double* trace_x, double* trace_time, int64_t iter_freq) {
    return prox_asaga(x, A_data, A_indices, A_indptr, b, d, n_samples, n_features,
      n_threads, alpha, beta, step_size, max_iter, trace_x, trace_time, iter_freq);
    }
}
