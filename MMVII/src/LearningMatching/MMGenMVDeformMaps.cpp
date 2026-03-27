#include "MMVII_all.h"
#include "V1VII.h"
#include "MMVII_Matrix.h"
#include "MMVII_Linear2DFiltering.h"

static int NODATA=-9999;


namespace  MMVII {

  namespace cNS_MGenDeformMaps
  {

    class cMGenDeformMaps;


    class cMGenDeformMaps: public cMMVII_Appli
    {
    public:
      typedef tU_INT1               tElemMasq;
      typedef tU_INT2               tElemImage;
      typedef tREAL4                tElemDepth;
      typedef cIm2D<tElemMasq>      tImMasq;
      typedef cIm2D<tElemImage>     tImImage;
      typedef cIm2D<tElemDepth>     tImDepth;
      typedef cDataIm2D<tElemMasq>  tDImMasq;
      typedef cDataIm2D<tElemImage> tDImImage;
      typedef cDataIm2D<tElemDepth> tDImDepth;
      typedef std::vector<tImImage>  tVecIms;
      typedef std::vector<tImMasq>   tVecMasq;
      cMGenDeformMaps(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
      cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
      cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
      int Exe() override;
      // GETTERS
      //std::vector<std::pair<std::string, std::string>> GetCouples() {return mCouplesImages;};
      // SETTERS
      //void SetCouples (std::vector<std::pair<std::string, std::string>> Couples ){mCouplesImages=Couples;}
      string NameImMasq(string NameIM);
      string NameImDepth(string NameIM);
      string NameImOri(string NameIM,std::string OriFolder, string SuffOri);
      string NameOut(string NameIm, string sfx);


      // --- constructed ---
      cPhotogrammetricProject   mPhProj;
      std::string  mNameIm1, mNameIm2;
      std::string mNameSuffixOut;
      cPt2di mSzIm;
      tImDepth mDx,mDy;
      tImMasq mMx,mMy;

    //private:
      //  std::vector<std::pair<std::string, std::string>> mCouplesImages;
    };


          // class definition
    cMGenDeformMaps::cMGenDeformMaps(const std::vector<string> &aVArgs, const cSpecMMVII_Appli &aSpec):
        cMMVII_Appli  (aVArgs,aSpec),
        mPhProj(*this),
        mSzIm(cPt2di(0,0)),
        mDx(cPt2di(1,1)),
        mDy(cPt2di(1,1)),
        mMx(cPt2di(1,1)),
        mMy(cPt2di(1,1))
    {
    }


    // Arg Mandatory
    cCollecSpecArg2007 & cMGenDeformMaps::ArgObl(cCollecSpecArg2007 &anArgObl)
    {
      return
           anArgObl
               <<   Arg2007(mNameIm1,"xml file from mm3d GrapheHom to get pair of viewing images")
               <<   Arg2007(mNameIm2,"xml file from mm3d GrapheHom to get pair of viewing images")
               <<   mPhProj.DPOrient().ArgDirInMand()
              // <<   Arg2007(mMasterImage,"Master image to project point cloud to it !")
               //<<   Arg2007(mPointCloud,"Point cloud in ply or las format !")
        ;
     }

    // Arg Optional
    cCollecSpecArg2007 & cMGenDeformMaps::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
         return anArgOpt
                << AOpt2007(mNameSuffixOut,"SuffixOut" ,"Default is defor_x and defor_y")
         ;
    }

    string cMGenDeformMaps::NameImDepth(string NameIM)
    {
      /*std::size_t fd=NameIM.find_last_of(".");
      string Depth=NameIM.substr(0,fd)+"_Pax1.tif";*/
    string Depth="DensifyPx_"+NameIM;
      return Depth;
    }
    string cMGenDeformMaps::NameOut(string NameIM, string sfx)
    {
      std::size_t fd=NameIM.find_last_of(".");
      return NameIM.substr(0,fd)+"_"+sfx+".tif";
    }

    string cMGenDeformMaps::NameImMasq(string NameIM)
    {
     /* size_t fd=NameIM.find_last_of(".");
      string Masq=NameIM.substr(0,fd)+"_Masq1.tif";*/
      std::string Masq="DensifyMasq_"+NameIM;
      return Masq;
    }

    string cMGenDeformMaps::NameImOri(string NameIM,std::string OriFolder, string SuffOri)
    {
      return OriFolder+"/"+SuffOri+NameIM+".xml";
    }


  int  cMGenDeformMaps::Exe()
    {
      std::string N1,N2,N1M,N2M;
      std::string aDirOris,aPatOri;
      N1=NameImDepth(mNameIm1);
      N1M=NameImMasq(mNameIm1);
      N2=NameImDepth(mNameIm2);
      N2M=NameImMasq(mNameIm2);


      cSensorCamPC * aCam1V2 = mPhProj.ReadCamPC(mNameIm1,true);
      cSensorCamPC * aCam2V2 = mPhProj.ReadCamPC(mNameIm2,true);


      mSzIm=aCam1V2->SzPix();

      // read images Depth and Masqs
      tImDepth aDepth1=tImDepth::FromFile(N1);
      tDImDepth & aDDepth1=aDepth1.DIm();
      tImDepth aDepth2=tImDepth::FromFile(N2);
      tDImDepth & aDDepth2=aDepth2.DIm();

      // Masq
      //tImMasq aMasq1=tImMasq::FromFile(N1M);
      //tDImMasq & aDDMasq1=aMasq1.DIm();

      //tImMasq aMasq2=tImMasq::FromFile(N2M);
      //tDImMasq & aDDMasq2=aMasq2.DIm();

      // Create Images Out Dx, Dy

      mDx=tImDepth(mSzIm,nullptr,eModeInitImage::eMIA_Null);
      //mDy=tImDepth(mSzIm,nullptr,eModeInitImage::eMIA_Null);
      mMx =tImMasq(mSzIm,nullptr,eModeInitImage::eMIA_Null);
      //mMy =tImMasq(mSzIm,nullptr,eModeInitImage::eMIA_Null);

      tDImDepth & aDxIm = mDx.DIm();
      aDxIm.InitCste(NODATA);
      //tDImDepth & aDyIm = mDy.DIm();
      //aDyIm.InitCste(NODATA);
      tDImMasq  & aMxIm= mMx.DIm();
      //tDImMasq  & aMyIm= mMy.DIm();

      // reproject row, col , depth from Im1 to Im2 and store Offset x and Offset y

      cPt2di aPix;
      for (aPix.y()=0;aPix.y()<mSzIm.y();aPix.y()++)
        {
          for (aPix.x()=0;aPix.x()<mSzIm.x();aPix.x()++)
            {
              if (1)//aDDMasq1.GetV(aPix)) // gradient mask -> may be we should discard it
                {
                  // DDEPTH CONTAINS GROUND TRUTH
                  // ADD perturbations to let locations in the image traval along epipolar lines
                    cPt3dr aPTer = aCam1V2->ImageAndDepth2Ground(cPt3dr(aPix.x(),aPix.y(),aDDepth1.GetV(aPix)));

                    if (aCam2V2->IsVisible(aPTer))
                      {
                        cPt2dr PtCam=aCam2V2->Ground2Image(aPTer);
                        // Check if we can get back to the first point
                        aDxIm.SetV(aPix,PtCam.x()-aPix.x());
                        //aDyIm.SetV(aPix,PtCam.y-aPix.y());

                        if (aDDepth2.InsideBL(PtCam))
                          {
                            cPt3dr aP3DIm2= aCam2V2->ImageAndDepth2Ground(cPt3dr(PtCam.x(),PtCam.y(),aDDepth2.GetVBL(PtCam)));
                            cPt2dr aP2InCam1=aCam1V2->Ground2Image(aP3DIm2);
                            cPt2dr aDiff= aP2InCam1-cPt2dr(aPix.x(),aPix.y());
                            if ((sqrt(aDiff.x()*aDiff.x()+aDiff.y()*aDiff.y()))<0.5) // Reprojection to the same point
                              {
                                // point is visible in both images
                                //cPt2di aIntP((int)PtCam.x(),(int)PtCam.y());
                                if(1)//aDDMasq2.GetV(aIntP))
                                  {
                                    aMxIm.SetV(aPix,1);
                                    //aMyIm.SetV(aPix,1);
                                  }
                              }
                          }

                      }
                }
            }
        }
        // dilate before saving
          auto  aMxImV1  = cMMV1_Conv<tU_INT1>::ImToMMV1(aMxIm);
          // dilate
          ELISE_COPY(aMxImV1.all_pts(),
                     dilat_d8(aMxImV1.in(0),4),
                     aMxImV1.out());

          // save Dx, Dy, Mx, My
          aDxIm.ToFile(NameOut(mNameIm1+"_"+mNameIm2,"Dx"));
          //aDyIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"Dy"));
          aMxIm.ToFile(NameOut(mNameIm1+"_"+mNameIm2,"Mx"));
          //aMyIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"My"));

      return EXIT_SUCCESS;
    }

  };


using namespace cNS_MGenDeformMaps;

  tMMVII_UnikPApli Alloc_MGenDeformMaps(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
  {
     return tMMVII_UnikPApli(new cMGenDeformMaps(aVArgs,aSpec));
  }

  cSpecMMVII_Appli TheSpecMGenDeformMaps
  (
       "DeformationMapsGivenDepths",
        Alloc_MGenDeformMaps,
        "Generate Flow Field using images orientations and depths",
        {eApF::Match},
        {eApDT::Image},
        {eApDT::ToDef},
        __FILE__
  );

};
