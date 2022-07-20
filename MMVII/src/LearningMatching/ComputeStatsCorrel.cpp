#include "include/MMVII_all.h"
//#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
//#include "include/MMVII_Tpl_Images.h"
#include <torch/torch.h>
#include <fstream>

/*********************************************************************/
template <typename T> void Tensor2File(torch::Tensor a, std::string fname, std::string Type)
{
   //Store Tensor 
   T * TensorContent=a.data_ptr<T>();
   FILE *finaldestination = fopen(fname.c_str(), "wb");
   fwrite(TensorContent, sizeof(T), a.numel(), finaldestination);
   fclose(finaldestination);
   //Store dimensions 
   std::string dimFname=fname;
   dimFname.append(".dim");
   std::ofstream finaldestinationDim(dimFname.c_str());
   
   for (int dd=0;dd<a.dim();dd++)
   {
      finaldestinationDim<<a.size(dd)<<std::endl;
   }
   finaldestinationDim.close();
   
   //Store data type 
   std::ofstream datatypetensor(fname.append(".type").c_str());
   datatypetensor<<Type<<std::endl;
   datatypetensor.close();
}

/*********************************************************************/
namespace MMVII
{
  //cAppliTestHypStep
class cAppliComputeStatsCorrel: public cAppliLearningMatch
{
     public :
        typedef cIm2D<tREAL4>              tImRad;   
        typedef cDataIm2D<tREAL4>          tDataImRad;

        cAppliComputeStatsCorrel(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        double ComputCorrel(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const;
        // -------------- Mandatory args -------------------
        std::string mNameI1;
        std::string mNameI2;
        std::string mNameImGT;
        int True1=1;
        int False1=2;
        int False2=8;
        int mPSzW=3;
        cBox2di BoxFile1() const {return cDataFileIm2D::Create(mNameI1,false);}
        cBox2di BoxFile2() const {return cDataFileIm2D::Create(mNameI2,false);}

        tImRad      mIm1;
        tDataImRad  *mDI1;
        tImRad      mIm2;
        tDataImRad  *mDI2;
        tImRad      mImDisp;
        tDataImRad  *mDIImDisp;
     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};

cAppliComputeStatsCorrel::cAppliComputeStatsCorrel(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cAppliLearningMatch  (aVArgs,aSpec),
   mIm1                 (cPt2di(1,1)),
   mDI1                 (nullptr),
   mIm2                 (cPt2di(1,1)),
   mDI2                 (nullptr),
   mImDisp              (cPt2di(1,1)),
   mDIImDisp            (nullptr)
   
{
}


double cAppliComputeStatsCorrel::ComputCorrel(const cPt2di & aPCI1,const cPt2dr & aPCI2,int aSzW) const
{
   cMatIner2Var<tREAL4> aMat;

   for (int aDx=-aSzW ; aDx<=aSzW  ; aDx++)
   {
       for (int aDy=-aSzW ; aDy<=aSzW  ; aDy++)
       {
            aMat.Add
            (
                mDI1->GetV  (aPCI1+cPt2di(aDx,aDy)),
                mDI2->GetVBL(aPCI2+cPt2dr(aDx,aDy))
            );
       }
   }

   return aMat.Correl();
}


cCollecSpecArg2007 & cAppliComputeStatsCorrel::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
 return
      anArgObl
          <<   Arg2007(mNameI1,"Name of first image")
          <<   Arg2007(mNameI2,"Name of second image")
          <<   Arg2007(mNameImGT,"Name of disparity image")
   ;
}

cCollecSpecArg2007 & cAppliComputeStatsCorrel::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return anArgOpt
           << AOpt2007(mPSzW, "WINSZ","corr window size",{eTA2007::HDV})
           << AOpt2007(True1, "True1","Positive Example offset",{eTA2007::HDV})
           << AOpt2007(False1, "False1","Negative Example first offset",{eTA2007::HDV})
           << AOpt2007(False2, "False2","Negative Example Second offset",{eTA2007::HDV})
   ;
}


int  cAppliComputeStatsCorrel::Exe()
{
   mIm1 = tImRad::FromFile(mNameI1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2);
   mDI2 = &(mIm2.DIm());
   mImDisp = tImRad::FromFile(mNameImGT);
   mDIImDisp = &(mImDisp.DIm());   
   
   cPt2di aSz = mDI1->Sz();
   //int mPSzW=7;
   cPt2di aSzW=cPt2di(mPSzW,mPSzW);
   std::vector<tREAL4> PositiveExamples,NegativeExamples;
   for (int aY=round_ni(mPSzW/2) ; aY<aSz.y()-round_ni(mPSzW/2) ; aY++)
   {
       for (int  aX =round_ni(mPSzW/2);aX<aSz.x()-round_ni(mPSzW/2) ; aX++)
       {
           // Compute correl between image Patches if Disparity is Not NAN 
           tREAL4 aValDisp=mDIImDisp->GetV(cPt2di(aX,aY));
           if (aValDisp != -999.0)
           {
                  // POSITTIVE RANDOM VALUE 
                  torch::Tensor d_posT=torch::randint(-True1,True1,{1});
                  tREAL4 dpos=d_posT.accessor<float,1>()[0];
                  // NEGATIVE RANDOM VALUE
                  torch::Tensor d_negT=torch::randint(False1,False2,{1});
                  tREAL4 dneg=d_negT.accessor<float,1>()[0];
                  torch::Tensor rr=torch::rand({1});
                  if (rr.accessor<float,1>()[0]<0.5)
                    {
                      dneg=-dneg;
                    }
                  
                  // GET OFFSET 
                  tREAL4 offpos_=aValDisp+dpos;
                  tREAL4 offneg_=aValDisp+dneg;
                  cPt2di aPC1(aX,aY);
                  cPt2dr aPC2P(aX-offpos_,aY);
                  cPt2dr aPC2N(aX-offneg_,aY);
                  if(WindInside4BL(this->BoxFile1(),aPC1,aSzW) && WindInside4BL(this->BoxFile2(),aPC2P,aSzW) && WindInside4BL(this->BoxFile2(),aPC2N,aSzW))
                    {
                      // COMPUTE CORREL POSITIVE AND NEGATIVE 
                        tREAL4 CORR_POS=this->ComputCorrel(aPC1,aPC2P,mPSzW);
                        tREAL4 CORR_NEG=this->ComputCorrel(aPC1,aPC2N,mPSzW);
                        PositiveExamples.push_back(CORR_POS);
                        NegativeExamples.push_back(CORR_NEG);
                    }
           }
        }
                
       }
      // DUMP POSITIVE AND NEGATIVE CORRELATIONS VALUES INTO torch Tensor  
      torch::Tensor aPOS=torch::from_blob(PositiveExamples.data(), {(int)PositiveExamples.size()},torch::TensorOptions().dtype(torch::kFloat32));
      torch::Tensor aNEG=torch::from_blob(NegativeExamples.data(), {(int)PositiveExamples.size()},torch::TensorOptions().dtype(torch::kFloat32));
      std::string pos=mNameI1+"_pos.bin";
      Tensor2File<float>(aPOS,pos,"float32");
      std::string neg=mNameI1+"_neg.bin";
      Tensor2File<float>(aNEG,neg,"float32");
      return EXIT_SUCCESS;
 }


tMMVII_UnikPApli Alloc_ComputeStatsCorrel(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliComputeStatsCorrel(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpecComputeStatsCorrel
(
     "ComputeCorrelStats",
      Alloc_ComputeStatsCorrel,
      "Compute statistics on correlation coefficients",
      {eApF::Match},
      {eApDT::Image},
      {eApDT::ToDef},
      __FILE__
);



};
