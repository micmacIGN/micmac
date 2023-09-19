#include "include/MMVII_all.h"
#include <StdAfx.h>

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

      std::string  mXmlCouplesImages;
      std::string mOriFolder;
      std::string mNameSuffixOut;
      bool mConik=true;
      cPt2di mSzIm;
      tImDepth mDx,mDy;
      tImMasq mMx,mMy;

    //private:
      //  std::vector<std::pair<std::string, std::string>> mCouplesImages;
    };


          // class definition
    cMGenDeformMaps::cMGenDeformMaps(const std::vector<string> &aVArgs, const cSpecMMVII_Appli &aSpec):
        cMMVII_Appli  (aVArgs,aSpec),
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
               <<   Arg2007(mXmlCouplesImages,"xml file from mm3d GrapheHom to get pair of viewing images")
               <<   Arg2007(mOriFolder,"Micmac Orientation Folder with Ori")
              // <<   Arg2007(mMasterImage,"Master image to project point cloud to it !")
               //<<   Arg2007(mPointCloud,"Point cloud in ply or las format !")
        ;
     }

    // Arg Optional
    cCollecSpecArg2007 & cMGenDeformMaps::ArgOpt(cCollecSpecArg2007 &anArgOpt)
    {
         return anArgOpt
                << AOpt2007(mNameSuffixOut,"SuffixOut" ,"Default is defor_x and defor_y")
                << AOpt2007(mConik,"Conik" ,"Geom capteur: default true")
         ;
    }

    string cMGenDeformMaps::NameImDepth(string NameIM)
    {
      std::size_t fd=NameIM.find_last_of(".");
      string Depth=NameIM.substr(0,fd)+"_Pax1.tif";
      return Depth;
    }
    string cMGenDeformMaps::NameOut(string NameIM, string sfx)
    {
      std::size_t fd=NameIM.find_last_of(".");
      return NameIM.substr(0,fd)+"_"+sfx+".tif";
    }

    string cMGenDeformMaps::NameImMasq(string NameIM)
    {
      size_t fd=NameIM.find_last_of(".");
      string Masq=NameIM.substr(0,fd)+"_Masq1.tif";
      return Masq;
    }

    string cMGenDeformMaps::NameImOri(string NameIM,std::string OriFolder, string SuffOri)
    {
      return OriFolder+"/"+SuffOri+NameIM+".xml";
    }

  int  cMGenDeformMaps::Exe()
    {
      std::string N1,N2,N1M,N2M,N1Ori,N2Ori;
      std::string aDirOris,aPatOri;
      SplitDirAndFile(aDirOris,aPatOri,mOriFolder,false);
     // load xml image pairs
      MMVII_INTERNAL_ASSERT_strong(mXmlCouplesImages!="", " Please give cple file as of mm3d GrapheHom !");
      cSauvegardeNamedRel aCpleSet = StdGetFromPCP(mXmlCouplesImages,SauvegardeNamedRel);
      cInterfChantierNameManipulateur * aICNMOris=cInterfChantierNameManipulateur::BasicAlloc(aDirOris);
      std::vector<cCpleString>::iterator itV=aCpleSet.Cple().begin();

      for (
           ;itV!=aCpleSet.Cple().end()
           ;itV++
           )
        {

          N1=NameImDepth(itV->N1());
          N1M=NameImMasq(itV->N1());
          N1Ori=NameImOri(itV->N1(),aDirOris,mConik ? "Orientation-":"GB-Orientation-:");
          N2=NameImDepth(itV->N2());
          N2M=NameImMasq(itV->N2());
          N2Ori=NameImOri(itV->N2(),aDirOris,mConik ? "Orientation-":"GB-Orientation-:");

          // get orientations
          CamStenope * aCam1 = CamOrientGenFromFile(N1Ori,aICNMOris);
          CamStenope * aCam2 = CamOrientGenFromFile(N2Ori,aICNMOris);

          mSzIm=cPt2di(aCam1->Sz().x,aCam1->Sz().y);

          // read images Depth and Masqs
          tImDepth aDepth1=tImDepth::FromFile(N1);
          tDImDepth & aDDepth1=aDepth1.DIm();
          //tImDepth aDepth2=tImDepth::FromFile(N2);
          //tDImDepth & aDDepth2=aDepth2.DIm();

          // Masq
          tImMasq aMasq1=tImMasq::FromFile(N1M);
          tDImMasq & aDDMasq1=aMasq1.DIm();

          tImMasq aMasq2=tImMasq::FromFile(N2M);
          tDImMasq & aDDMasq2=aMasq2.DIm();

          // Create Images Out Dx, Dy

          mDx=tImDepth(mSzIm,nullptr,eModeInitImage::eMIA_Null);
          mDy=tImDepth(mSzIm,nullptr,eModeInitImage::eMIA_Null);
          mMx =tImMasq(mSzIm,nullptr,eModeInitImage::eMIA_Null);
          mMy =tImMasq(mSzIm,nullptr,eModeInitImage::eMIA_Null);

          tDImDepth & aDxIm = mDx.DIm();
          aDxIm.InitCste(NODATA);
          tDImDepth & aDyIm = mDy.DIm();
          aDyIm.InitCste(NODATA);
          tDImMasq  & aMxIm= mMx.DIm();
          tDImMasq  & aMyIm= mMy.DIm();

          // reproject row, col , depth from Im1 to Im2 and store Offset x and Offset y

          cPt2di aPix;
          for (aPix.y()=0;aPix.y()<mSzIm.y();aPix.y()++)
            {
              for (aPix.x()=0;aPix.x()<mSzIm.x();aPix.x()++)
                {
                  if (aDDMasq1.GetV(aPix))
                    {
                        Pt3dr aTer=aCam1->ImEtProf2Terrain(Pt2dr(aPix.x(),aPix.y()),aDDepth1.GetV(aPix));

                        if (aCam2->PIsVisibleInImage(aTer))
                          {
                            Pt2dr PtCam=aCam2->Ter2Capteur(aTer);
                            cPt2di aIntP((int)PtCam.x,(int)PtCam.y);
                            if(aDDMasq2.GetV(aIntP))
                              {
                                aMxIm.SetV(aPix,1);
                                aMyIm.SetV(aPix,1);
                                aDxIm.SetV(aPix,PtCam.x);
                                aDyIm.SetV(aPix,PtCam.y);
                              }
                          }
                    }

                }

            }

          // save Dx, Dy, Mx, My
          aDxIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"Dx"));
          aDyIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"Dy"));
          aMxIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"Mx"));
          aMyIm.ToFile(NameOut(itV->N1()+"__"+itV->N2(),"My"));
        }

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
