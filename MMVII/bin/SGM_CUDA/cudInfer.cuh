#pragma once
#include <stdio.h>
#include <assert.h>
#include <math_constants.h>
#include <stdint.h>
#include <unistd.h>
#include <torch/torch.h>


__device__ void sort(float *x, int n);
__global__ void cross(float *x0, float *out, int size, int dim2, int dim3, int L1, float tau1);
__global__ void cbca(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction);
__global__ void cbcaFull(float *x0c, float *x1c, float *vol, float *out, int size, int dim2, int dim3, int direction,int lowerrange);
template <int sgm_direction> __global__ void sgm2(float *x0, float *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step);
template <int sgm_direction> __global__ void sgm2Full(float *x0, float *x1, float *input, float *output, float *tmp, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int size1, int size2, int size3, int step, int lowerrange);
__global__ void outlier_detection(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max);
__global__ void outlier_detectionFull(float *d0, float *d1, float *outlier, int size, int dim3, int disp_max, int disp_min);
__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3);
__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max);
__global__ void subpixel_enchancementFull(float *d0, float *c2, float *out, int size, int dim23, int disp_max, int disp_min);
__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2);
__global__ void StereoJoin_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23);
__global__ void StereoJoinFull_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23,int lowerrange);
__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius);

void sgm2(torch::Tensor x0, torch::Tensor x1, torch::Tensor input , torch::Tensor output, torch::Tensor tmp,
     float pi1,float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction
        );
void sgm2Full(torch::Tensor x0, torch::Tensor x1, torch::Tensor input , torch::Tensor output, torch::Tensor tmp,
     float pi1,float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int direction, int LowerRange
        );       
void CrBaCoAgg(torch::Tensor x0c, torch::Tensor x1c, torch::Tensor vol_in, torch::Tensor vol_out,  int direction);
void CrBaCoAggFull(torch::Tensor x0c, torch::Tensor x1c, torch::Tensor vol_in, torch::Tensor vol_out,  int direction,int LowerRange);
void Cross(torch::Tensor x0, torch::Tensor out, int L1, float tau1);
void checkCudaError();
void outlier_detection (torch::Tensor d0, torch::Tensor d1, torch::Tensor outlier, int disp_max);
void outlier_detectionFull (torch::Tensor d0, torch::Tensor d1, torch::Tensor outlier, int disp_max, int disp_min);
void interpolate_mismatch(torch::Tensor d0, torch::Tensor outlier, torch::Tensor out);
void interpolate_occlusion(torch::Tensor d0, torch::Tensor outlier,torch::Tensor out);
void subpixel_enchancement(torch::Tensor d0, torch::Tensor c2, torch::Tensor out, int disp_max) ;
void subpixel_enchancementFull(torch::Tensor d0, torch::Tensor c2, torch::Tensor out, int disp_max, int disp_min) ;
void mean2d(torch::Tensor img, torch::Tensor kernel, torch::Tensor out, float alpha2);
int StereoJoin(torch::Tensor input_L, torch::Tensor input_R, torch::Tensor output_L,torch::Tensor output_R);
int StereoJoinFull(torch::Tensor input_L, torch::Tensor input_R, torch::Tensor output_L,torch::Tensor output_R,int LowerRange);
void median2d(torch::Tensor img, torch::Tensor out, int kernel_size);









