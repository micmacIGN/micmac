// File Automatically generated by eLiSe
#include "StdAfx.h"
#include "cEqObsBaseGPS.h"


cEqObsBaseGPS::cEqObsBaseGPS():
    cElCompiledFonc(3)
{
   AddIntRef (cIncIntervale("Base",6,9));
   AddIntRef (cIncIntervale("Orient",0,6));
   Close(false);
}



void cEqObsBaseGPS::ComputeVal()
{
   double tmp0_ = mCompCoord[0];
   double tmp1_ = cos(tmp0_);
   double tmp2_ = mCompCoord[1];
   double tmp3_ = mCompCoord[2];
   double tmp4_ = sin(tmp0_);
   double tmp5_ = -(tmp4_);
   double tmp6_ = sin(tmp3_);
   double tmp7_ = sin(tmp2_);
   double tmp8_ = -(tmp7_);
   double tmp9_ = tmp1_*tmp8_;
   double tmp10_ = cos(tmp3_);
   double tmp11_ = cos(tmp2_);
   double tmp12_ = mCompCoord[6];
   double tmp13_ = mCompCoord[7];
   double tmp14_ = -(tmp6_);
   double tmp15_ = tmp4_*tmp8_;
   double tmp16_ = mCompCoord[8];

  mVal[0] = (tmp1_*tmp11_*tmp12_+(tmp5_*tmp10_+tmp9_*tmp6_)*tmp13_+(tmp5_*tmp14_+tmp9_*tmp10_)*tmp16_+mCompCoord[3])-mLocGPS_x;

  mVal[1] = (tmp4_*tmp11_*tmp12_+(tmp1_*tmp10_+tmp15_*tmp6_)*tmp13_+(tmp1_*tmp14_+tmp15_*tmp10_)*tmp16_+mCompCoord[4])-mLocGPS_y;

  mVal[2] = (tmp7_*tmp12_+tmp11_*tmp6_*tmp13_+tmp11_*tmp10_*tmp16_+mCompCoord[5])-mLocGPS_z;

}


void cEqObsBaseGPS::ComputeValDeriv()
{
   double tmp0_ = mCompCoord[0];
   double tmp1_ = cos(tmp0_);
   double tmp2_ = mCompCoord[1];
   double tmp3_ = mCompCoord[2];
   double tmp4_ = sin(tmp0_);
   double tmp5_ = -(tmp4_);
   double tmp6_ = sin(tmp3_);
   double tmp7_ = sin(tmp2_);
   double tmp8_ = -(tmp7_);
   double tmp9_ = tmp1_*tmp8_;
   double tmp10_ = cos(tmp3_);
   double tmp11_ = cos(tmp2_);
   double tmp12_ = mCompCoord[6];
   double tmp13_ = -(1);
   double tmp14_ = tmp13_*tmp4_;
   double tmp15_ = mCompCoord[7];
   double tmp16_ = -(tmp1_);
   double tmp17_ = -(tmp6_);
   double tmp18_ = tmp14_*tmp8_;
   double tmp19_ = mCompCoord[8];
   double tmp20_ = -(tmp11_);
   double tmp21_ = tmp20_*tmp1_;
   double tmp22_ = tmp13_*tmp6_;
   double tmp23_ = tmp1_*tmp11_;
   double tmp24_ = tmp5_*tmp10_;
   double tmp25_ = tmp9_*tmp6_;
   double tmp26_ = tmp24_+tmp25_;
   double tmp27_ = tmp5_*tmp17_;
   double tmp28_ = tmp9_*tmp10_;
   double tmp29_ = tmp27_+tmp28_;
   double tmp30_ = tmp4_*tmp8_;
   double tmp31_ = tmp23_*tmp12_;
   double tmp32_ = tmp13_*tmp7_;
   double tmp33_ = tmp20_*tmp4_;
   double tmp34_ = -(tmp10_);
   double tmp35_ = tmp4_*tmp11_;
   double tmp36_ = tmp1_*tmp10_;
   double tmp37_ = tmp30_*tmp6_;
   double tmp38_ = tmp36_+tmp37_;
   double tmp39_ = tmp1_*tmp17_;
   double tmp40_ = tmp30_*tmp10_;
   double tmp41_ = tmp39_+tmp40_;
   double tmp42_ = tmp11_*tmp6_;
   double tmp43_ = tmp11_*tmp10_;

  mVal[0] = (tmp31_+(tmp26_)*tmp15_+(tmp29_)*tmp19_+mCompCoord[3])-mLocGPS_x;

  mCompDer[0][0] = tmp14_*tmp11_*tmp12_+(tmp16_*tmp10_+tmp18_*tmp6_)*tmp15_+(tmp16_*tmp17_+tmp18_*tmp10_)*tmp19_;
  mCompDer[0][1] = tmp32_*tmp1_*tmp12_+tmp21_*tmp6_*tmp15_+tmp21_*tmp10_*tmp19_;
  mCompDer[0][2] = (tmp22_*tmp5_+tmp10_*tmp9_)*tmp15_+(tmp34_*tmp5_+tmp22_*tmp9_)*tmp19_;
  mCompDer[0][3] = 1;
  mCompDer[0][4] = 0;
  mCompDer[0][5] = 0;
  mCompDer[0][6] = tmp23_;
  mCompDer[0][7] = tmp26_;
  mCompDer[0][8] = tmp29_;
  mVal[1] = (tmp35_*tmp12_+(tmp38_)*tmp15_+(tmp41_)*tmp19_+mCompCoord[4])-mLocGPS_y;

  mCompDer[1][0] = tmp31_+(tmp14_*tmp10_+tmp25_)*tmp15_+(tmp14_*tmp17_+tmp28_)*tmp19_;
  mCompDer[1][1] = tmp32_*tmp4_*tmp12_+tmp33_*tmp6_*tmp15_+tmp33_*tmp10_*tmp19_;
  mCompDer[1][2] = (tmp22_*tmp1_+tmp10_*tmp30_)*tmp15_+(tmp34_*tmp1_+tmp22_*tmp30_)*tmp19_;
  mCompDer[1][3] = 0;
  mCompDer[1][4] = 1;
  mCompDer[1][5] = 0;
  mCompDer[1][6] = tmp35_;
  mCompDer[1][7] = tmp38_;
  mCompDer[1][8] = tmp41_;
  mVal[2] = (tmp7_*tmp12_+tmp42_*tmp15_+tmp43_*tmp19_+mCompCoord[5])-mLocGPS_z;

  mCompDer[2][0] = 0;
  mCompDer[2][1] = tmp11_*tmp12_+tmp32_*tmp6_*tmp15_+tmp32_*tmp10_*tmp19_;
  mCompDer[2][2] = tmp10_*tmp11_*tmp15_+tmp22_*tmp11_*tmp19_;
  mCompDer[2][3] = 0;
  mCompDer[2][4] = 0;
  mCompDer[2][5] = 1;
  mCompDer[2][6] = tmp7_;
  mCompDer[2][7] = tmp42_;
  mCompDer[2][8] = tmp43_;
}


void cEqObsBaseGPS::ComputeValDerivHessian()
{
  ELISE_ASSERT(false,"Foncteur cEqObsBaseGPS Has no Der Sec");
}

void cEqObsBaseGPS::SetGPS_x(double aVal){ mLocGPS_x = aVal;}
void cEqObsBaseGPS::SetGPS_y(double aVal){ mLocGPS_y = aVal;}
void cEqObsBaseGPS::SetGPS_z(double aVal){ mLocGPS_z = aVal;}



double * cEqObsBaseGPS::AdrVarLocFromString(const std::string & aName)
{
   if (aName == "GPS_x") return & mLocGPS_x;
   if (aName == "GPS_y") return & mLocGPS_y;
   if (aName == "GPS_z") return & mLocGPS_z;
   return 0;
}


cElCompiledFonc::cAutoAddEntry cEqObsBaseGPS::mTheAuto("cEqObsBaseGPS",cEqObsBaseGPS::Alloc);


cElCompiledFonc *  cEqObsBaseGPS::Alloc()
{  return new cEqObsBaseGPS();
}


