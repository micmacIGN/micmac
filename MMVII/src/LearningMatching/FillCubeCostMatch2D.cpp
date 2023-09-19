#include "include/MMVII_all.h"
#include "include/MMVII_2Include_Serial_Tpl.h"
#include "LearnDM.h"
#include "include/MMVII_Tpl_Images.h"
#include "include/MMVII_TplLayers3D.h"
#include <thread>


// included model cnn
#include "cCnnModelPredictor.h"

const int SLICE_DIV=2;

namespace MMVII {

  namespace  cNS_FillCubeCost2D {

  class cAppliFillCubeCost2D;

  static const std::string TheUnetMlpCubeMatcher="UnetMLPMatcher";

  class cAppliFillCubeCost2D : public cAppliLearningMatch
  {
       public :
          typedef tINT2                      tElemZ;
          typedef cIm2D<tElemZ>              tImZ;
          typedef cDataIm2D<tElemZ>          tDataImZ;
          typedef cIm2D<tREAL4>              tImRad;
          typedef cDataIm2D<tREAL4>          tDataImRad;
          typedef cLayer3D<float,tElemZ>     tLayerCor;
          typedef cIm2D<tREAL4>              tImPx;
          typedef cDataIm2D<tU_INT1>         tDataImMasq;
          typedef cDataIm2D<tREAL4>          tDataImPx;
          typedef cIm2D<tREAL4>              tImFiltred;
          typedef cDataIm2D<tREAL4>          tDataImF;
          typedef cGaussianPyramid<tREAL4>   tPyr;
          typedef std::shared_ptr<tPyr>      tSP_Pyr;
          typedef cPyr1ImLearnMatch *        tPtrPyr1ILM;

          cAppliFillCubeCost2D(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
          /*************************************************************************/
          const tDataImRad & DI1() {return *mDI1;}
          const tDataImRad & DI2() {return *mDI2;}
          /*************************************************************************/
          tREAL8     StepZ() const {return mStepZ;}
          const cBox2di  & BoxGlob1() const {return mBoxGlob1;}  ///< Accessor
          const cBox2di  & BoxGlob2() const {return mBoxGlob2;}  ///< Accessor
          const std::string   & NameI1() const {return mNameI1;}  ///< Accessor
          const std::string   & NameI2() const {return mNameI2;}  ///< Accessor

          const std::string  & NameArch() const {return mModelArchitecture;} // ACCESSOR
          const std::string  & NameDirModel() const {return mModelBinaries;} // ACCESSOR

          cBox2di BoxFile1() const {return cDataFileIm2D::Create(mNameI1,false);}
          cBox2di BoxFile2() const {return cDataFileIm2D::Create(mNameI2,false);}
          const tImZ  & ImZMin1() {return  mImZMin1;}
          const tImZ  & ImZMax1() {return  mImZMax1;}

          const tImZ  & ImZMin2() {return  mImZMin2;}
          const tImZ  & ImZMax2() {return  mImZMax2;}
          torch::Tensor InterpolateSlice(torch::Tensor & FeatMap,torch::Tensor & aGeoXT,torch::Tensor & aGeoYT);
          void ExeOptim();
          aCnnModelPredictor * mCNNPredictor=nullptr;
          bool mWithPredictionNetwork=false;
          bool mWithMatcher3D=false;
          torch::jit::script::Module mMSNet;
          torch::jit::script::Module mDecisionNet;
          torch::jit::script::Module mMatcherNet;
    private:
          int Exe() override;
          cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
          cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

          void PushCost(double aCost);
          // -------------- Mandatory args -------------------
          std::string   mNameI1;
          std::string   mNameI2;
          std::string   mNameModele;
          cPt2di        mP0Z;  // Pt corresponding in Im1 to (0,0)
          cBox2di       mBoxGlob1;  // Box to Load, taking into account siwe effect
          cBox2di       mBoxGlob2;
          std::string   mNamePost;

          // -------------- Optionnal args -------------------
          tREAL8        mStepZ;
          std::string   mNameCmpModele;

          // ADDED CNN PARAMS
          std::string mModelBinaries;
          std::string mModelArchitecture;

          // -------------- Internal variables -------------------

          std::string StdName(const std::string & aPre,const std::string & aPost);

          int         mNbCmpCL;
          cIm2D<tREAL8>  mImCmp;

          // Pax 1
          std::string mNameZMin1;
          std::string mNameZMax1;
          // Pax 2
          std::string mNameZMin2;
          std::string mNameZMax2;

          std::string mNameCube;
          cMMVII_Ofs* mFileCube;

          tImZ        mImZMin1;
          tImZ        mImZMax1;

          tImZ        mImZMin2;
          tImZ        mImZMax2;

          tImRad      mIm1;
          tDataImRad  *mDI1;
          tImRad      mIm2;
          tDataImRad  *mDI2;
          double      ToCmpCost(double aCost) const;
   };

  cAppliFillCubeCost2D::cAppliFillCubeCost2D(const std::vector<std::string> &aVArgs, const cSpecMMVII_Appli &aSpec) :
    cAppliLearningMatch   (aVArgs,aSpec),
    mBoxGlob1             (cBox2di::Empty()),
    mBoxGlob2             (cBox2di::Empty()),
    mStepZ                (1.0),
    mNbCmpCL              (200),
    mImCmp                (cPt2di(mNbCmpCL+2,mNbCmpCL+2),nullptr,eModeInitImage::eMIA_Null),
    mFileCube             (nullptr),
    mImZMin1              (cPt2di(1,1)),
    mImZMax1              (cPt2di(1,1)),
    mImZMin2              (cPt2di(1,1)),
    mImZMax2              (cPt2di(1,1)),
    mIm1                  (cPt2di(1,1)),
    mDI1                  (nullptr),
    mIm2                  (cPt2di(1,1)),
    mDI2                  (nullptr)
 {
 }

  cCollecSpecArg2007 & cAppliFillCubeCost2D::ArgObl(cCollecSpecArg2007 & anArgObl)
  {
   return
        anArgObl
            <<   Arg2007(mNameI1,"Name of first image")
            <<   Arg2007(mNameI2,"Name of second image")
            <<   Arg2007(mNameModele,"Name for modele :  "+TheUnetMlpCubeMatcher)
            <<   Arg2007(mP0Z,"Origin in first image")
            <<   Arg2007(mBoxGlob1,"Box to read 4 Im1")
            <<   Arg2007(mBoxGlob2,"Box to read 4 Im2")
            <<   Arg2007(mNamePost,"Post fix for other names (ZMin1,ZMax1,ZMin2,ZMax2,Cube)")
     ;
  }

  cCollecSpecArg2007 & cAppliFillCubeCost2D::ArgOpt(cCollecSpecArg2007 & anArgOpt)
  {
     return anArgOpt
            << AOpt2007(mStepZ, "StepZ","Step for paralax",{eTA2007::HDV})
            << AOpt2007(mNameCmpModele, "ModCmp","Modele for Comparison")
            << AOpt2007(mModelBinaries,"CNNParams" ,"Model Directory : scripted model files *.pt")
            << AOpt2007(mModelArchitecture,"CNNArch" ,"Model architecture : " + TheUnetMlpCubeMatcher)
     ;
  }

  std::string cAppliFillCubeCost2D::StdName(const std::string & aPre,const std::string & aPost)
  {
          return aPre + "_" + mNamePost + "." + aPost;
  }

  void cAppliFillCubeCost2D::PushCost(double aCost)
  {
     tU_INT2 aICost = round_ni(1e4*(std::max(0.0,std::min(1.0,aCost))));
     mFileCube->Write(aICost);
  }

torch::Tensor cAppliFillCubeCost2D::InterpolateSlice(torch::Tensor & aFeatMap, torch::Tensor & aGeoXT,torch::Tensor & aGeoYT)
{
  //oversampling near a location given the predefined step aStep: sub-pixel level
  using namespace torch::indexing;
  auto HFeatMap=aFeatMap.size(1);
  auto WFeatMap=aFeatMap.size(2);

  // Tensor Interpolate at once
  torch::Tensor anInterPolFeatMap=torch::zeros({aFeatMap.size(0),aGeoXT.size(0),aGeoXT.size(1)},
                                               torch::TensorOptions().dtype(torch::kFloat32));
  // Get definition Tensors
  auto DEF_X=torch::mul((aGeoXT>=0), (aGeoXT<WFeatMap));
  auto DEF_Y=torch::mul((aGeoYT>=0), (aGeoYT<HFeatMap));

  // floor and ceil
  torch::Tensor Y_1=torch::floor(aGeoYT).mul(DEF_Y);
  torch::Tensor X_1=torch::floor(aGeoXT).mul(DEF_X);

  torch::Tensor Y_2=torch::ceil(aGeoYT).mul(DEF_Y);
  torch::Tensor X_2=torch::ceil(aGeoXT).mul(DEF_X);

  aGeoXT=aGeoXT.mul(DEF_X);
  aGeoYT=aGeoYT.mul(DEF_Y);

  Y_2=Y_2.mul(Y_2<HFeatMap);
  X_2=X_2.mul(X_2<WFeatMap);
  //auto aMAP_11= torch::gather(alongx,-1,Y_1.unsqueeze(0).repeat_interleave(FeatSz,0).to(torch::kInt64));
  //auto aMAP_11=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_1.to(torch::kInt64)})});
  auto aMAP_11=aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_1.to(torch::kInt64)});
  //std::cout<<"  composed offsets "<<aMAP_11.sizes()<<std::endl;
  //auto aMAP_21=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_1.to(torch::kInt64)})});
  auto aMAP_21=aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_1.to(torch::kInt64)});
  //auto aMAP_12=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_2.to(torch::kInt64)})});
  auto aMAP_12=aFeatMap.index({Slice(),Y_1.to(torch::kInt64),X_2.to(torch::kInt64)});
  //auto aMAP_22=torch::einsum("ijk->ijk",{aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_2.to(torch::kInt64)})});
  auto aMAP_22=aFeatMap.index({Slice(),Y_2.to(torch::kInt64),X_2.to(torch::kInt64)});
  //std::cout<<"  MAP 22 DIMENSIONS ------   X"<<aMAP_22.sizes()<<"   Y  "<<aMAP_22.sizes()<<std::endl;

  /*bool oktest1=true;
  if (oktest1)
      {
          // Tests on feature warping routine correctness
          auto aFeat= aMAP_11.index({Slice(0,None,1),50,50});
          int y_ind=(int) Y_1.index({50,50}).item<double>();
          int x_ind=(int) X_1.index({50,50}).item<double>();
          auto aFeatFromIndices=aFeatMap.index({Slice(0,None,1),y_ind,x_ind});
          auto isEqual=torch::equal(aFeat,aFeatFromIndices);
          MMVII_INTERNAL_ASSERT_strong(isEqual, "PROBLEM WITH FEATURE WARPIGN BEFORE INTERPOLATION !!!");
      }*/

  // INTERPOLATE THE WHOLE FEATURE MAP

  //             InterpValue= ((double)x_2-aLocX)*(aMap11*y_2_y+aMap21*y_y_1)
  //              + (aLocX-(double)x_1)*(aMap12*y_2_y+aMap22*y_y_1);


//                               |aMAP11 aMAP21| |Y_2 - Y|
//   out = |X_2 - X , X - X_1 |  |             | |       |
//                               |aMAP12 aMAP22| |Y - Y_1|

  auto Y_1GEOYT=aGeoYT-Y_1;
  auto Y_2GEOYT=Y_2-aGeoYT;
  auto X_1GEOXT=aGeoXT-X_1;
  auto X_2GEOXT=X_2-aGeoXT;
  Y_2GEOYT.index_put_({Y_2GEOYT==0},1);
  X_2GEOXT.index_put_({X_2GEOXT==0},1);
  auto TERM_1= torch::einsum("ijk,jk->ijk",{aMAP_11,Y_2GEOYT})+torch::einsum("ijk,jk->ijk",{aMAP_21,Y_1GEOYT});
  auto TERM_2= torch::einsum("ijk,jk->ijk",{aMAP_12,Y_2GEOYT})+torch::einsum("ijk,jk->ijk",{aMAP_22,Y_1GEOYT});
  anInterPolFeatMap=torch::einsum("ijk,jk->ijk",{TERM_1,X_2GEOXT})+torch::einsum("ijk,jk->ijk",{TERM_2,X_1GEOXT});

  return anInterPolFeatMap;
}

void cAppliFillCubeCost2D::ExeOptim()
{
  // Load temporary data outout by MMV1

   // Compute names and load images of Pax1 and Pax2 priors
   // Pax1
   mNameZMin1 = StdName("ZMin1","tif");
   mNameZMax1 = StdName("ZMax1","tif");

   // Pax2
   mNameZMin2 = StdName("ZMin2","tif");
   mNameZMax2 = StdName("ZMax2","tif");

   mNameCube = StdName("MatchingCube","data");
   // Read images
   mImZMin1 = tImZ::FromFile(mNameZMin1);
   tDataImZ & aDZMin1 = mImZMin1.DIm();
   mImZMax1 = tImZ::FromFile(mNameZMax1);
   tDataImZ & aDZMax1 = mImZMax1.DIm();

   // Pax 2
   mImZMin2 = tImZ::FromFile(mNameZMin2);
   tDataImZ & aDZMin2 = mImZMin2.DIm();
   mImZMax2 = tImZ::FromFile(mNameZMax2);
   tDataImZ & aDZMax2 = mImZMax2.DIm();

   mIm1 = tImRad::FromFile(mNameI1,mBoxGlob1);
   mDI1 = &(mIm1.DIm());
   mIm2 = tImRad::FromFile(mNameI2,mBoxGlob2);
   mDI2 = &(mIm2.DIm());

   mFileCube = new cMMVII_Ofs(mNameCube,false);


   // Call Models and Fill Cost Cube : This time we have 2D disparity search ranges
   //bool cuda_available= torch::cuda::is_available();
   //torch::Device Device(cuda_available ? torch::kCUDA : torch::kCPU);
   torch::Device Device=torch::kCPU;
   // Calculate
   cPt2di aSzL = this->DI1().Sz();
   cPt2di aSzR = this->DI2().Sz();
   cPt2di aPix;
   torch::Tensor Feat_reference,Feat_query;
   // LOADING MODELS
   if (NameArch()==TheUnetMlpCubeMatcher)
       {
           mCNNPredictor = new aCnnModelPredictor(TheUnetMlpCubeMatcher,this->NameDirModel());
           mCNNPredictor->PopulateModelFeatures(mMSNet);
           if (mWithPredictionNetwork)
           {
             mCNNPredictor->PopulateModelDecision(mDecisionNet);
           }

           if (mWithMatcher3D)
               {
                   // Enhance the  generated correlation coefficients using the last stage conv3d MATCHER
                    mCNNPredictor->PopulateModelMatcher(mMatcherNet);
               }

           // get Embeddings from the feature extractor

           Feat_reference=mCNNPredictor->PredictMSNetTileFeatures(mMSNet,this->mIm1,aSzL);
           Feat_query=mCNNPredictor->PredictMSNetTileFeatures(mMSNet,this->mIm2,aSzR);

           // Fill the Cost Volume with 2D search interval given the per pixel limts in Zinf and Zsup

           // to device if cuda


           Feat_query=Feat_query.to(Device).unsqueeze(0) ;
           Feat_reference=Feat_reference.to(Device).squeeze();
           // Interpolate query features according to Step()

           std::cout<<"  "<<Feat_reference.sizes()<<"  "<<Feat_query.sizes()<<std::endl;

           namespace F= torch::nn::functional;
           /*******************************************************************************/
           /*******************************************************************************/
           /*******  Simple Loop solution --> memory overflow *****************************/
           /*******************************************************************************/
           /*******************************************************************************/
           // initial checks
           int aZMin = 1e9;
           int aZMax = -1e9;
           {
               for (int aX=0 ; aX<aSzL.x() ; aX++)
               {
                   for (int aY=0 ; aY<aSzL.y() ; aY++)
                   {
                         aZMin=std::min(aZMin,(int)aDZMin1.GetV(cPt2di(aX,aY)));
                         aZMax=std::max(aZMax,(int)aDZMax1.GetV(cPt2di(aX,aY)));
                   }
               }
               aZMin = round_down(aZMin*StepZ());
               aZMax = round_up  (aZMax*StepZ());
           }
           std::cout<<"BORNES INITIALES "<<aZMin<<"   "<<aZMax<<std::endl;
           aZMin = 1e9;
           aZMax = -1e9;
           {
               for (int aX=0 ; aX<aSzL.x() ; aX++)
               {
                   for (int aY=0 ; aY<aSzL.y() ; aY++)
                   {
                         aZMin=std::min(aZMin,(int)aDZMin2.GetV(cPt2di(aX,aY)));
                         aZMax=std::max(aZMax,(int)aDZMax2.GetV(cPt2di(aX,aY)));
                   }
               }
               aZMin = round_down(aZMin*StepZ());
               aZMax = round_up  (aZMax*StepZ());
           }
           std::cout<<"BORNES INITIALES PAX 2 "<<aZMin<<"   "<<aZMax<<std::endl;
           /*******************************************************************************/
           /*******************************************************************************/
           /******  Optimized solution to solve sub-pixel interpolation memory overhead ***/
           /*******************************************************************************/
           /*******************************************************************************/
           if (StepZ()<1.0)
             {
               using namespace torch::indexing;
               std::vector<double> new_scale_factor{1.0/StepZ(),1.0/StepZ()};
               int WIDTHbyDIV =(int)aSzL.x()/SLICE_DIV ;
               int HEIGHTbyDIV=(int)aSzL.y()/SLICE_DIV ;
               int Reste_WIDTH=aSzL.x()%SLICE_DIV ;
               int Reste_HEIGHT=aSzL.y()%SLICE_DIV;

               for (int div_y=0; div_y<SLICE_DIV;div_y++)
                 {
                   for (int div_x=0;div_x<SLICE_DIV;div_x++)
                     {
                       // Calculer les bornes des nappes englobantes pour savoir decouper la feature map  et interpoler
                       double ZMIN_1=1e9;
                       double ZMIN_2=1e9;
                       double ZMAX_1=-1e9;
                       double ZMAX_2=-1e9;
                       int SUPP_X= (div_x==SLICE_DIV-1) ? Reste_WIDTH:0;
                       int SUPP_Y= (div_y==SLICE_DIV-1) ? Reste_HEIGHT:0;
                       for (aPix.y()=div_y*HEIGHTbyDIV; aPix.y()<(div_y+1)*HEIGHTbyDIV+SUPP_Y;aPix.y()++)
                         {
                           for (aPix.x()=div_x*WIDTHbyDIV; aPix.x()<(div_x+1)*WIDTHbyDIV+SUPP_X;aPix.x()++)
                             {
                               cPt2di aPC20 = aPix + mP0Z-mBoxGlob2.P0();
                               ZMIN_1=std::min(ZMIN_1,aPC20.x()+aDZMin1.GetV(aPix)*StepZ());
                               ZMIN_2=std::min(ZMIN_2,aPC20.y()+aDZMin2.GetV(aPix)*StepZ());
                               ZMAX_1=std::max(ZMAX_1,aPC20.x()+aDZMax1.GetV(aPix)*StepZ());
                               ZMAX_2=std::max(ZMAX_2,aPC20.y()+aDZMax2.GetV(aPix)*StepZ());
                             }
                         }
                       std::cout<<"REEL "<<ZMIN_1<<"  "<<ZMAX_1<<"  "<<ZMIN_2<<"   "<<ZMAX_2<<std::endl;
                       // bornes par rapport à limage ?
                       int ZMIN_1N=round_down(std::max(ZMIN_1,0.0));
                       int ZMIN_2N=round_down(std::max(ZMIN_2,0.0));
                       int ZMAX_1N=round_up(std::min(ZMAX_1,(double)Feat_query.size(-1)));
                       int ZMAX_2N=round_up(std::min(ZMAX_2,(double)Feat_query.size(-2)));
                       //

                       std::cout<<"ARRONDI  "<<ZMIN_2N<<"  "<<ZMAX_2N<<"  "<<ZMIN_1N<<"   "<<ZMAX_1N<<std::endl;

                       torch::Tensor FeatQuerySlice=Feat_query.index({Slice(0,None,1),
                                                                      Slice(0,None,1),
                                                                      Slice(ZMIN_2N,ZMAX_2N,1),
                                                                      Slice(ZMIN_1N,ZMAX_1N,1)});   // a tester lors de lappel aux indices

                       std::cout<<"SLICE SIZES "<<FeatQuerySlice.sizes()<<std::endl;
                       //Interpolate sub pixel
                       FeatQuerySlice=F::interpolate(FeatQuerySlice,
                                                     F::InterpolateFuncOptions().scale_factor(new_scale_factor).mode(torch::kBilinear)).squeeze();

                       std::cout<<" FeatQuerySlice Intepolated "<<FeatQuerySlice.sizes()<<std::endl;

                       int EpaisseurNappe_1, EpaisseurNappe_2;

                       for (aPix.y()=div_y*HEIGHTbyDIV; aPix.y()<(div_y+1)*HEIGHTbyDIV+SUPP_Y;aPix.y()++)
                         {
                           for (aPix.x()=div_x*WIDTHbyDIV; aPix.x()<(div_x+1)*WIDTHbyDIV+SUPP_X;aPix.x()++)
                             {
                               cPt2di aPAbs = aPix + mP0Z;
                               cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                               cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                               // compute similarity scores for each position given the search intervals
                               EpaisseurNappe_1=(int)(aDZMax1.GetV(aPix)-aDZMin1.GetV(aPix));
                               EpaisseurNappe_2=(int)(aDZMax2.GetV(aPix)-aDZMin2.GetV(aPix));

                               torch::Tensor All_Reference=torch::zeros({Feat_reference.size(0),EpaisseurNappe_2, EpaisseurNappe_1},
                                                                         torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                                        );

                               torch::Tensor All_Query=torch::zeros({FeatQuerySlice.size(0),EpaisseurNappe_2, EpaisseurNappe_1},
                                                                     torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                                    );
                               // Fill All_Reference and All_Query
                               // Reference
                               auto aRef=Feat_reference.index({Slice(0,None,1),aPC1.y(),aPC1.x()}).unsqueeze(1).unsqueeze(2);
                               aRef=aRef.expand(All_Reference.sizes()) ;
                               All_Reference.copy_(aRef);

                               int inf_x,inf_y,sup_x,sup_y;
                               inf_x=(int)(aPC20.x()/StepZ()+aDZMin1.GetV(aPix)-ZMIN_1/StepZ());// /StepZ();
                               sup_x=(int)(aPC20.x()/StepZ()+aDZMax1.GetV(aPix)-ZMIN_1/StepZ());// /StepZ();
                               inf_y=(int)(aPC20.y()/StepZ()+aDZMin2.GetV(aPix)-ZMIN_2/StepZ());// /StepZ();
                               sup_y=(int)(aPC20.y()/StepZ()+aDZMax2.GetV(aPix)-ZMIN_2/StepZ());// /StepZ();
                               int _inf_x, _sup_x, _inf_y, _sup_y;
                               _inf_x= (inf_x<0) ? -inf_x:0;
                               _inf_y= (inf_y<0) ? -inf_y:0;
                               _sup_x= (sup_x>FeatQuerySlice.size(2)) ? All_Query.size(2)-(sup_x-FeatQuerySlice.size(2)):All_Query.size(2);
                               _sup_y= (sup_y>FeatQuerySlice.size(1)) ? All_Query.size(1)-(sup_y-FeatQuerySlice.size(1)):All_Query.size(1);
                               //std::cout<<_inf_y<<"   "<<_sup_y<<"  "<<_inf_x<<"  "<<_sup_x<<std::endl;
                               //std::cout<<inf_y<<"   "<<sup_y<<"  "<<inf_x<<"  "<<sup_x<<std::endl;

                               inf_x=std::max(inf_x,0);
                               sup_x=std::min(sup_x,(int)FeatQuerySlice.size(2));
                               inf_y=std::max(inf_y,0);
                               sup_y=std::min(sup_y,(int)FeatQuerySlice.size(1));
                               All_Query.index({Slice(0,None,1),
                                                Slice(_inf_y,_sup_y,1),
                                                Slice(_inf_x,_sup_x,1)}).copy_(FeatQuerySlice.index({Slice(0,None,1),
                                                                     Slice(inf_y,sup_y,1),
                                                                     Slice(inf_x,sup_x,1)}));

                               // compute similarity
                               //std::cout<<" All_Query SIZES  "<< All_Query.sizes()<<std::endl;
                               torch::Tensor CosSim=F::cosine_similarity(All_Reference,
                                                                               All_Query,
                                                                               F::CosineSimilarityFuncOptions().dim(0)).squeeze();
                               for (int aDzy=aDZMin2.GetV(aPix) ; aDzy<aDZMax2.GetV(aPix) ; aDzy++)
                               {
                                   for (int aDzx=aDZMin1.GetV(aPix) ; aDzx<aDZMax1.GetV(aPix) ; aDzx++)
                                     {
                                         double aTabCost[2]={1.0,1.0};
                                         //bool   aTabOk[2]={false,false};
                                         cPt2dr aPC2Z(aPC20.x()+aDzx*StepZ(),aPC20.y()+aDzy*StepZ());
                                         bool IsInside=WindInside4BL(this->DI1(),aPC1,cPt2di(1,1)) && WindInside4BL(this->DI2(),aPC2Z,cPt2di(1,1));
                                         if (IsInside)
                                           {
                                             auto aSim=CosSim.index({(int64_t)(aDzy-aDZMin2.GetV(aPix)),
                                                                           (int64_t)(aDzx-aDZMin1.GetV(aPix))});
                                             ELISE_ASSERT(aSim.item<float>()<=1.0 && aSim.item<float>()>=-1.0, "Similarity values issue not in bound 0 ,1 ");
                                             aTabCost[0] =(1-(double)aSim.item<float>())/2.0;
                                           }
                                           PushCost(aTabCost[0]);
                                     }
                               }
                             }
                         }
                     }
                 }
             }
       }
   else
     {
       std::cerr<<"Do Not Consider Any Other Model: JUST UnetMLPMatcher\n";
     }

   delete mFileCube;
}

 int cAppliFillCubeCost2D::Exe()
  {
   // Load temporary data outout by MMV1

    // Compute names and load images of Pax1 and Pax2 priors
    // Pax1
    mNameZMin1 = StdName("ZMin1","tif");
    mNameZMax1 = StdName("ZMax1","tif");

    // Pax2
    mNameZMin2 = StdName("ZMin2","tif");
    mNameZMax2 = StdName("ZMax2","tif");

    mNameCube = StdName("MatchingCube","data");
    // Read images
    mImZMin1 = tImZ::FromFile(mNameZMin1);
    tDataImZ & aDZMin1 = mImZMin1.DIm();
    mImZMax1 = tImZ::FromFile(mNameZMax1);
    tDataImZ & aDZMax1 = mImZMax1.DIm();

    // Pax 2
    mImZMin2 = tImZ::FromFile(mNameZMin2);
    tDataImZ & aDZMin2 = mImZMin2.DIm();
    mImZMax2 = tImZ::FromFile(mNameZMax2);
    tDataImZ & aDZMax2 = mImZMax2.DIm();

    mIm1 = tImRad::FromFile(mNameI1,mBoxGlob1);
    mDI1 = &(mIm1.DIm());
    mIm2 = tImRad::FromFile(mNameI2,mBoxGlob2);
    mDI2 = &(mIm2.DIm());

    mFileCube = new cMMVII_Ofs(mNameCube,false);


    // Call Models and Fill Cost Cube : This time we have 2D disparity search ranges
    //bool cuda_available= torch::cuda::is_available();
    //torch::Device Device(cuda_available ? torch::kCUDA : torch::kCPU);
    torch::Device Device=torch::kCPU;
    // Calculate
    cPt2di aSzL = this->DI1().Sz();
    cPt2di aSzR = this->DI2().Sz();
    cPt2di aPix;
    torch::Tensor Feat_reference,Feat_query;
    // LOADING MODELS
    if (NameArch()==TheUnetMlpCubeMatcher)
        {
            mCNNPredictor = new aCnnModelPredictor(TheUnetMlpCubeMatcher,this->NameDirModel());
            mCNNPredictor->PopulateModelFeatures(mMSNet);
            if (mWithPredictionNetwork)
            {
              mCNNPredictor->PopulateModelDecision(mDecisionNet);
            }

            if (mWithMatcher3D)
                {
                    // Enhance the  generated correlation coefficients using the last stage conv3d MATCHER
                     mCNNPredictor->PopulateModelMatcher(mMatcherNet);
                }

            // get Embeddings from the feature extractor

            Feat_reference=mCNNPredictor->PredictMSNetTileFeatures(mMSNet,this->mIm1,aSzL);
            Feat_query=mCNNPredictor->PredictMSNetTileFeatures(mMSNet,this->mIm2,aSzR);

            // Fill the Cost Volume with 2D search interval given the per pixel limts in Zinf and Zsup
            // to device if cuda
            Feat_query=Feat_query.to(Device).squeeze() ;
            Feat_reference=Feat_reference.to(Device).squeeze();
            std::cout<<"  "<<Feat_reference.sizes()<<"  "<<Feat_query.sizes()<<std::endl;

            namespace F= torch::nn::functional;
            /*******************************************************************************/
            /*******************************************************************************/
            /*******  Simple Loop solution --> memory overflow *****************************/
            /*******************************************************************************/
            /*******************************************************************************/
            // Compute similarity
            int EpaisseurNappe_1, EpaisseurNappe_2;
            using namespace torch::indexing;
            for (aPix.y()=0; aPix.y()<aSzL.y();aPix.y()++)
              {
                for (aPix.x()=0; aPix.x()<aSzL.x();aPix.x()++)
                  {
                    cPt2di aPAbs = aPix + mP0Z;
                    cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                    cPt2di aPC20 = aPAbs-mBoxGlob2.P0();

                    // compute similarity scores for each position given the search intervals
                    EpaisseurNappe_1=(int)(aDZMax1.GetV(aPix)-aDZMin1.GetV(aPix));
                    EpaisseurNappe_2=(int)(aDZMax2.GetV(aPix)-aDZMin2.GetV(aPix));

                    torch::Tensor All_Reference=torch::zeros({Feat_reference.size(0),EpaisseurNappe_2, EpaisseurNappe_1},
                                                              torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                             );

                    torch::Tensor All_Query=torch::zeros({Feat_query.size(0),EpaisseurNappe_2, EpaisseurNappe_1},
                                                          torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                         );

                    // Fill All_Reference and All_Query
                    // Reference
                    auto aRef=Feat_reference.index({Slice(0,None,1),aPC1.y(),aPC1.x()}).unsqueeze(1).unsqueeze(2);

                    aRef=aRef.expand(All_Reference.sizes()) ;
                    All_Reference.copy_(aRef);
                    int inf_x,inf_y,sup_x,sup_y;
                    bool ByPASS=true;
                    if (ByPASS)
                      {
                        if (StepZ()<1.0)
                          {
                            torch::Tensor aGeoX=torch::zeros({EpaisseurNappe_2,EpaisseurNappe_1},
                                                             torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                             );
                            torch::Tensor aGeoY=torch::zeros({EpaisseurNappe_2,EpaisseurNappe_1},
                                                             torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                             );
                            // fill
                            for (int aDzy=aDZMin2.GetV(aPix) ; aDzy<aDZMax2.GetV(aPix) ; aDzy++)
                            {
                                for (int aDzx=aDZMin1.GetV(aPix) ; aDzx<aDZMax1.GetV(aPix) ; aDzx++)
                                  {
                                      cPt2dr aPC2Z(aPC20.x()+aDzx*StepZ(),aPC20.y()+aDzy*StepZ());
                                      aGeoX.index({(int64_t)(aDzy-aDZMin2.GetV(aPix)),
                                                   (int64_t)(aDzx-aDZMin1.GetV(aPix))}).copy_(torch::tensor(aPC2Z.x()));
                                      aGeoY.index({(int64_t)(aDzy-aDZMin2.GetV(aPix)),
                                                  (int64_t)(aDzx-aDZMin1.GetV(aPix))}).copy_(torch::tensor(aPC2Z.y()));

                                  }
                             }
                            // interpolate
                            All_Query=this->InterpolateSlice(Feat_query,aGeoX,aGeoY);
                          }
                        else if (StepZ()==1.0)
                          {
                            inf_x=(int)(aPC20.x()+aDZMin1.GetV(aPix));
                            sup_x=(int)(aPC20.x()+aDZMax1.GetV(aPix));
                            inf_y=(int)(aPC20.y()+aDZMin2.GetV(aPix));
                            sup_y=(int)(aPC20.y()+aDZMax2.GetV(aPix));
                            int _inf_x, _sup_x, _inf_y, _sup_y;
                            _inf_x= (inf_x<0) ? -inf_x:0;
                            _inf_y= (inf_y<0) ? -inf_y:0;
                            _sup_x= (sup_x>Feat_query.size(2)) ? All_Query.size(2)-(sup_x-Feat_query.size(2)):All_Query.size(2);
                            _sup_y= (sup_y>Feat_query.size(1)) ? All_Query.size(1)-(sup_y-Feat_query.size(1)):All_Query.size(1);
                            //std::cout<<_inf_y<<"   "<<_sup_y<<"  "<<_inf_x<<"  "<<_sup_x<<std::endl;
                            //std::cout<<inf_y<<"   "<<sup_y<<"  "<<inf_x<<"  "<<sup_x<<std::endl;

                            inf_x=std::max(inf_x,0);
                            sup_x=std::min(sup_x,(int)Feat_query.size(2));
                            inf_y=std::max(inf_y,0);
                            sup_y=std::min(sup_y,(int)Feat_query.size(1));
                            All_Query.index({Slice(0,None,1),
                                             Slice(_inf_y,_sup_y,1),
                                             Slice(_inf_x,_sup_x,1)}).copy_(Feat_query.index({Slice(0,None,1),
                                                                  Slice(inf_y,sup_y,1),
                                                                  Slice(inf_x,sup_x,1)}));
                          }
                        else
                          {
                            // nothing to do

                          }
                      }
                    else
                      {
                          inf_x=(int)(aPC20.x()/StepZ()+aDZMin1.GetV(aPix));// /StepZ();
                          sup_x=(int)(aPC20.x()/StepZ()+aDZMax1.GetV(aPix));// /StepZ();
                          inf_y=(int)(aPC20.y()/StepZ()+aDZMin2.GetV(aPix));// /StepZ();
                          sup_y=(int)(aPC20.y()/StepZ()+aDZMax2.GetV(aPix));// /StepZ();
                          int _inf_x, _sup_x, _inf_y, _sup_y;
                          _inf_x= (inf_x<0) ? -inf_x:0;
                          _inf_y= (inf_y<0) ? -inf_y:0;
                          _sup_x= (sup_x>Feat_query.size(2)) ? All_Query.size(2)-(sup_x-Feat_query.size(2)):All_Query.size(2);
                          _sup_y= (sup_y>Feat_query.size(1)) ? All_Query.size(1)-(sup_y-Feat_query.size(1)):All_Query.size(1);
                          //std::cout<<_inf_y<<"   "<<_sup_y<<"  "<<_inf_x<<"  "<<_sup_x<<std::endl;
                          //std::cout<<inf_y<<"   "<<sup_y<<"  "<<inf_x<<"  "<<sup_x<<std::endl;

                          inf_x=std::max(inf_x,0);
                          sup_x=std::min(sup_x,(int)Feat_query.size(2));
                          inf_y=std::max(inf_y,0);
                          sup_y=std::min(sup_y,(int)Feat_query.size(1));
                          All_Query.index({Slice(0,None,1),
                                           Slice(_inf_y,_sup_y,1),
                                           Slice(_inf_x,_sup_x,1)}).copy_(Feat_query.index({Slice(0,None,1),
                                                                Slice(inf_y,sup_y,1),
                                                                Slice(inf_x,sup_x,1)}));
                      }


                    // compute similarity
                    //std::cout<<" All_Query SIZES  "<< All_Query.sizes()<<std::endl;
                    torch::Tensor CosSim=F::cosine_similarity(All_Reference,
                                                                    All_Query,
                                                                    F::CosineSimilarityFuncOptions().dim(0)).squeeze();
                    //std::cout<<" CosSim SIZES  "<< CosSim.sizes()<<std::endl;
                    // dump in mFileCube
                    for (int aDzy=aDZMin2.GetV(aPix) ; aDzy<aDZMax2.GetV(aPix) ; aDzy++)
                    {
                        for (int aDzx=aDZMin1.GetV(aPix) ; aDzx<aDZMax1.GetV(aPix) ; aDzx++)
                          {
                              double aTabCost[2]={0.5,1.0};
                              //bool   aTabOk[2]={false,false};
                              cPt2dr aPC2Z(aPC20.x()+aDzx*StepZ(),aPC20.y()+aDzy*StepZ());
                              bool IsInside=WindInside4BL(this->DI1(),aPC1,cPt2di(1,1)) && WindInside4BL(this->DI2(),aPC2Z,cPt2di(1,1));
                              if (IsInside)
                                {
                                  auto aSim=CosSim.index({(int64_t)(aDzy-aDZMin2.GetV(aPix)),
                                                                (int64_t)(aDzx-aDZMin1.GetV(aPix))});
                                  //ELISE_ASSERT(aSim.item<float>()<=1.0 && aSim.item<float>()>=-1.0, "Similarity values issue not in bound 0 ,1 ");
                                  //aTabCost[0] =(1-(double)aSim.item<float>())/2.0;
                                  double aCorrelMin=0.7;
                                  double aGamaCorr=2.0;
                                  double aRes =  ((double)aSim.item<float>()- aCorrelMin) / (1-aCorrelMin);  // 1->1 , CorrelMi->0
                                  aRes = std::pow(aRes,aGamaCorr);
                                  aTabCost[0]=aRes;
                                }
                                PushCost(aTabCost[0]);
                          }
                    }
                }
           }
            // Compute similarity
            /*int EpaisseurNappe_1=(int)(aDZMax1.GetV(cPt2di(0,0))-aDZMin1.GetV(cPt2di(0,0)));
            int EpaisseurNappe_2=(int)(aDZMax2.GetV(cPt2di(0,0))-aDZMin2.GetV(cPt2di(0,0)));

            std::cout<<EpaisseurNappe_1<<"    "<<EpaisseurNappe_2<<std::endl;

            EpaisseurNappe_1=(int)(aDZMax1.GetV(cPt2di(50,50))-aDZMin1.GetV(cPt2di(50,50)));
            EpaisseurNappe_2=(int)(aDZMax2.GetV(cPt2di(50,50))-aDZMin2.GetV(cPt2di(50,50)));
            std::cout<<EpaisseurNappe_1<<"    "<<EpaisseurNappe_2<<std::endl;

            // tensor of queries and references should have the size {FeatSize,EpaisseurNappe_2/STEP, EpaisseurNappe_1/STEP, Height, Width}

            MMVII_INTERNAL_ASSERT_strong(EpaisseurNappe_1>0, "Epaisseur Nappe Should be strictly positive !");
            MMVII_INTERNAL_ASSERT_strong(EpaisseurNappe_2>0, "Epaisseur Nappe transverse should be strictly positive !");

            torch::Tensor All_Reference=torch::zeros({Feat_reference.size(0),EpaisseurNappe_1, EpaisseurNappe_2,
                                                      Feat_reference.size(1), Feat_reference.size(2)},
                                                      torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                     );

            torch::Tensor All_Query=torch::zeros({Feat_query.size(0),EpaisseurNappe_1, EpaisseurNappe_2,
                                                  Feat_query.size(1), Feat_query.size(2)},
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(Device)
                                                 );
            using namespace torch::indexing;
            for (aPix.y()=0; aPix.y()<aSzL.y();aPix.y()++)
              {
                for (aPix.x()=0; aPix.x()<aSzL.x();aPix.x()++)
                  {
                    // Gather
                    //auto aGeoX=torch::arange(aDZMin2.GetV(aPix),aDZMax2.GetV(aPix),
                    //                         torch.TensorOptions().dtype(torch::kInt64)).expand();
                    cPt2di aPAbs = aPix + mP0Z;
                    cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                    cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                    auto aRef=Feat_reference.index({aPC1.y(),aPC1.x()});
                    aRef=aRef.expand({-1,All_Reference.size(1), All_Reference.size(2)});
                    All_Reference.index({Slice(0,None,1),Slice(0,None,1),Slice(0,None,1),
                                         aPix.y(),aPix.x()}).copy_(aRef);

                    All_Query.index({Slice(0,None,1),Slice(0,None,1),Slice(0,None,1),
                                     aPix.y(),aPix.x()}).copy_(Feat_query.index({Slice(0,None,1),
                                                                                Slice((int64_t)((aPC20.y()+aDZMin1.GetV(aPix))/StepZ()),
                                                                                 (int64_t)((aPC20.y()+aDZMax1.GetV(aPix))/StepZ()),1),
                                                                                Slice((int64_t)((aPC20.x()+aDZMin2.GetV(aPix))/StepZ()),
                                                                                 (int64_t)((aPC20.x()+aDZMax2.GetV(aPix))/StepZ()),1),
                                                                                aPix.y(),
                                                                                aPix.x()
                                                                                }));
                  }
              }
            // Compute the cosine similarity

            torch::Tensor aWholeCosSim=F::cosine_similarity(All_Reference, All_Query,F::CosineSimilarityFuncOptions().dim(0)).squeeze();
            // Dump in the mFileCube that needs to be restructured for 2D search intervals
            for (aPix.y()=0; aPix.y()<aSzL.y();aPix.y()++)
              {
                for (aPix.x()=0; aPix.x()<aSzL.x();aPix.x()++)
                  {
                      cPt2di aPAbs = aPix + mP0Z;
                      cPt2di aPC1  = aPAbs-mBoxGlob1.P0();
                      cPt2di aPC20 = aPAbs-mBoxGlob2.P0();
                      for (int aDzy=aDZMin1.GetV(aPix)/StepZ() ; aDzy<aDZMax1.GetV(aPix)/StepZ() ; aDzy++)
                      {
                          for (int aDzx=aDZMin2.GetV(aPix)/StepZ() ; aDzx<aDZMax2.GetV(aPix)/StepZ() ; aDzx++)
                            {
                              double aTabCost[2]={1.0,1.0};
                              //bool   aTabOk[2]={false,false};
                              cPt2dr aPC2Z(aPC20.x()+aDzx*StepZ(),aPC20.y()+aDzy*StepZ());
                              bool IsInside=WindInside4BL(this->DI1(),aPC1,cPt2di(3,3)) && WindInside4BL(this->DI2(),aPC2Z,cPt2di(3,3));
                              if (IsInside)
                                {
                                  // Fill cost cube
                                  auto aSim=aWholeCosSim.index({(int64_t)(aDzy-aDZMin1.GetV(aPix)/StepZ()),
                                                                (int64_t)(aDzx-aDZMin2.GetV(aPix)/StepZ()),
                                                                aPix.y(),
                                                                aPix.x()});
                                  ELISE_ASSERT(aSim.item<float>()<=1.0 && aSim.item<float>()>=0, "Similarity values issue not in bound 0 ,1 ");
                                  aTabCost[0] =1-(double)aSim.item<float>();
                                  //aTabOk[0]=true;
                                }
                              PushCost(aTabCost[0]);

                            }
                      }

                  }
              } */


        }
    else
      {
        std::cerr<<"Do Not Consider Any Other Model: JUST UnetMLPMatcher\n";
      }

    delete mFileCube;
    return EXIT_SUCCESS;
  }
 };
  /* =============================================== */
  /*                                                 */
  /*                       ::                        */
  /*                                                 */
  /* =============================================== */
  using namespace  cNS_FillCubeCost2D;

  tMMVII_UnikPApli Alloc_FillCubeCost2D(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
  {
     return tMMVII_UnikPApli(new cAppliFillCubeCost2D(aVArgs,aSpec));
  }

  cSpecMMVII_Appli  TheSpecFillCubeCost2D
  (
       "DM4FillCubeCost2D",
        Alloc_FillCubeCost2D,
        "Fill a cube with matching costs in a 2D search space",
        {eApF::Match},
        {eApDT::Image},
        {eApDT::ToDef},
        __FILE__
  );


};
