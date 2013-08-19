/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr

   
    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in 
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte 
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
#include "StdAfx.h"


using namespace NS_ParamChantierPhotogram;

class cImaMM;
class cAppliWithSetImage;
class cAppliMMByPair;


class cImaMM
{
    public :
      cImaMM(const std::string & aName,cAppliWithSetImage &);


    public :
       std::string mNameIm;
       std::string mBande;
       int         mNumInBande;
       CamStenope * mCam;
};

class cAppliWithSetImage
{
   public :
      CamStenope * CamOfName(const std::string & aName);
   protected :
      cAppliWithSetImage(int argc,char ** argv);
      void MakeStripStruct(const std::string & aPairByStrip,bool StripFirst);
      void VerifAWSI();
      void ComputeStripPair(int);
      void AddPair(cImaMM * anI1,cImaMM * anI2);

      bool        mSym;
      bool        mShow;
      std::string mPb;
      std::string mFullName;
      std::string mDir;
      std::string mPat;
      std::string mOri;
      std::string mKeyOri;
      cInterfChantierNameManipulateur * mICNM;
      const cInterfChantierNameManipulateur::tSet * mSetIm;

      std::vector<cImaMM *> mImages;
      typedef std::pair<cImaMM *,cImaMM *> tPairIm;
      typedef std::set<tPairIm> tSetPairIm;
      tSetPairIm   mPairs;


   private :
      void AddPairASym(cImaMM * anI1,cImaMM * anI2);
     
};





class cAppliMMByPair : public cAppliWithSetImage
{
    public :
      cAppliMMByPair(int argc,char ** argv);
      int Exe();
    private :
      void DoCorrel();
      void DoMDT();
      void DoBascule();
      void DoFusion();

      int mZoom0;
      int mZoomF;
      int mDiffInStrip;
      bool mStripIsFirt;
      std::string  mPairByStrip;
      std::string  mDirBasc;
      int          mNbStep;
};

/*****************************************************************/
/*                                                               */
/*                            cImaMM                             */
/*                                                               */
/*****************************************************************/

cImaMM::cImaMM(const std::string & aName,cAppliWithSetImage & anAppli) :
   mNameIm     (aName),
   mBande      (""),
   mNumInBande (-1),
   mCam        (anAppli.CamOfName(mNameIm))
{
}

/*****************************************************************/
/*                                                               */
/*                      cAppliWithSetImage                       */
/*                                                               */
/*****************************************************************/

static std::string aBlank(" ");

cAppliWithSetImage::cAppliWithSetImage(int argc,char ** argv)  :
   mSym  (true),
   mShow (false),
   mPb   ("")
{
   if (argc<2)
   {
      mPb = "Not Enough Arg in cAppliWithSetImage";
      return;
   }


   mFullName = argv[0];
#if (ELISE_windows)
        replace( mFullName.begin(), mFullName.end(), '\\', '/' );
#endif
   SplitDirAndFile(mDir,mPat,mFullName);

   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetIm = mICNM->Get(mPat);

   mOri = argv[1];
   mKeyOri =  "NKS-Assoc-Im2Orient@-" + mOri;

   for (int aKV=0 ; aKV<int(mSetIm->size()) ; aKV++)
   {
       mImages.push_back(new cImaMM((*mSetIm)[aKV],*this));
   }
}

void cAppliWithSetImage::VerifAWSI()
{
   ELISE_ASSERT(mPb=="",mPb.c_str());
}

CamStenope * cAppliWithSetImage::CamOfName(const std::string & aNameIm)
{
   std::string aNameOri =  mICNM->Assoc1To1(mKeyOri,aNameIm,true);
   return   CamOrientGenFromFile(aNameOri,mICNM);
}

void  cAppliWithSetImage::MakeStripStruct(const std::string & aPairByStrip,bool StripIsFirst)
{

  cElRegex anAutom(aPairByStrip.c_str(),10);
  std::string aExpStrip = StripIsFirst ? "$1" : "$2";
  std::string aExpNumInStrip = StripIsFirst ? "$2" : "$1";

  for (int aKI=0;  aKI<int(mImages.size()) ; aKI++)
  {
      cImaMM & anI = *(mImages[aKI]);
      
      std::string aBande = MatchAndReplace(anAutom,anI.mNameIm,aExpStrip);
      std::string aNumInBande = MatchAndReplace(anAutom,anI.mNameIm,aExpNumInStrip);

      bool OkNum = FromString(anI.mNumInBande,aNumInBande);
      ELISE_ASSERT(OkNum,"Num in bande is not numeric");
      if (mShow)
         std::cout << " Strip " << anI.mNameIm << " " << aBande <<  ";;" << anI.mNumInBande << "\n";
      anI.mBande = aBande;
  }
}


void cAppliWithSetImage::ComputeStripPair(int aDif)
{
    for (int aK1=0 ; aK1<int(mImages.size()) ; aK1++)
    {
        cImaMM & anI1 = *(mImages[aK1]);
        for (int aK2=0 ; aK2<int(mImages.size()) ; aK2++)
        {
            cImaMM & anI2 = *(mImages[aK2]);
            if (anI1.mBande==anI2.mBande)
            {
               int aN1 = anI1.mNumInBande;
               int aN2 = anI2.mNumInBande;
               if ((aN1>aN2) && (aN1<=aN2+aDif))
               {
                    AddPair(&anI1,&anI2);
               }
            }
        }
    }
}

void cAppliWithSetImage::AddPair(cImaMM * anI1,cImaMM * anI2)
{
    if (anI1>anI2) 
       ElSwap(anI1,anI2);
    AddPairASym(anI1,anI2);
    if (mSym)
       AddPairASym(anI2,anI1);
}

void cAppliWithSetImage::AddPairASym(cImaMM * anI1,cImaMM * anI2)
{
    tPairIm aPair(anI1,anI2);

    if (mPairs.find(aPair) != mPairs.end())
       return;
    
    mPairs.insert(aPair);

    if (mShow)
       std::cout << "Add Pair " << anI1->mNameIm << " " << anI2->mNameIm << "\n";
}



/*****************************************************************/
/*                                                               */
/*                                                               */
/*                                                               */
/*****************************************************************/

cAppliMMByPair::cAppliMMByPair(int argc,char ** argv) :
    cAppliWithSetImage (argc-1,argv+1),
    mZoom0       (64),
    mZoomF       (1),
    mDiffInStrip (1),
    mStripIsFirt (true),
    mDirBasc     ("MTD-Nuage")
{
  ElInitArgMain
  (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Name (Dir+Pattern)")
                    << EAMC(mOri,"Orientation"),
        LArgMain()  << EAM(mZoom0,"Zoom0",true,"Zoom Init, Def=64")
                    << EAM(mZoomF,"ZoomF",true,"Zoom Final, Def=1")
                    << EAM(mPairByStrip,"ByStrip",true,"Pair in same strip[Pat,ExprStrip,ExprNumInStrip]")
                    << EAM(mStripIsFirt,"StripIsFisrt",true,"If true : first expr is strip, second is num in strip")
                    << EAM(mDiffInStrip,"DeltaStrip",true,"Delta in same strip (Def=1,apply with mPairByStrip)")
                    << EAM(mSym,"Sym",true,"Symetrise all pair (Def=true)")
                    << EAM(mShow,"Show",true,"Show details (def = false))")
  );
  VerifAWSI();

  if (EAMIsInit(&mPairByStrip))
  {
      MakeStripStruct(mPairByStrip,mStripIsFirt);
      ComputeStripPair(mDiffInStrip);
  }

  mNbStep = round_ni(log2(mZoom0/double(mZoomF))) + 3 ;
}


void cAppliMMByPair::DoCorrel()
{
   for ( tSetPairIm::const_iterator itP= mPairs.begin(); itP!=mPairs.end() ; itP++)
   {
        cImaMM & anI1 = *(itP->first);
        cImaMM & anI2 = *(itP->second);

        std::string aCom =    MMBinFile("MICMAC")
                           +  XML_MM_File("MM-Param2Im.xml")
                           +  std::string(" WorkDir=") + mDir          + aBlank
                           +  std::string(" +Ori=") + mOri + aBlank
                           +  std::string(" +Im1=")    + anI1.mNameIm  + aBlank
                           +  std::string(" +Im2=")    + anI2.mNameIm  + aBlank
                           +  std::string(" +Zoom0=")  + ToString(mZoom0)  + aBlank
                           +  std::string(" +ZoomF=")  + ToString(mZoomF)  + aBlank
                         ;


        if (mShow)
           std::cout << aCom << "\n";
        System(aCom);
   }
}

void cAppliMMByPair::DoBascule()
{
   for ( tSetPairIm::const_iterator itP= mPairs.begin(); itP!=mPairs.end() ; itP++)
   {
        cImaMM & anI1 = *(itP->first);
        cImaMM & anI2 = *(itP->second);
        std::string aCom =    MMBinFile(MM3DStr) + " NuageBascule "
                             + mDir+ "MEC2Im-" + anI1.mNameIm + "-" +  anI2.mNameIm + "/NuageImProf_LeChantier_Etape_" +ToString(mNbStep)+".xml "
                             + mDir + mDirBasc + "/NuageImProf_LeChantier_Etape_1.xml "
                             + mDir + mDirBasc +  "/Basculed-"+ anI1.mNameIm + "-" + anI2.mNameIm + " "
                             
                            ;
        if (mShow)
           std::cout  << aCom << "\n";
        System(aCom);
   }
}

void cAppliMMByPair::DoFusion()
{
    std::string aCom =    MMBinFile(MM3DStr) + " MergeDepthMap "
                       +   XML_MM_File("Fusion-MMByP.xml") + aBlank
                       +   "  WorkDirPFM=" + mDir + mDirBasc + "/ ";
    if (mShow)
       std::cout  << aCom << "\n";
    System(aCom);
}

#define StdGetFromPCP(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("ParamChantierPhotogram.xml"),\
        #aObj ,\
        #aObj \
     )

#define StdGetFromSI(aStr,aObj)\
StdGetObjFromFile<c##aObj>\
    (\
    aStr,\
        StdGetFileXMLSpec("SuperposImage.xml"),\
        #aObj ,\
        #aObj \
     )


void cAppliMMByPair::DoMDT()
{
   std::string aCom =     MMBinFile("MICMAC")
                       +  XML_MM_File("MM-GenMTDNuage.xml")
                       +  std::string(" WorkDir=") + mDir          + aBlank
                       +  " +PatternAllIm=" +  mPat + aBlank
                       +  std::string(" +Ori=") + mOri + aBlank
                       +  std::string(" +Zoom=")  + ToString(mZoomF)  + aBlank
                       +  std::string(" +DirMEC=")  + mDirBasc  + aBlank
                    ;

   System(aCom);
 
   std::string aStrN = mDir+mDirBasc+"/NuageImProf_LeChantier_Etape_1.xml";
   cXML_ParamNuage3DMaille aNuage = StdGetFromSI(aStrN,XML_ParamNuage3DMaille);
   aNuage.PN3M_Nuage().Image_Profondeur().Val().OrigineAlti() = 0;
   aNuage.PN3M_Nuage().Image_Profondeur().Val().ResolutionAlti() = 1;
   MakeFileXML(aNuage,aStrN);



   std::string aStrZ = mDir+mDirBasc+"/Z_Num1_DeZoom"+ToString(mZoomF)+ "_LeChantier.xml";
   cFileOriMnt aFileZ = StdGetFromPCP(aStrZ,FileOriMnt);
   aFileZ.OrigineAlti() = 0;
   aFileZ.ResolutionAlti() = 1;
   MakeFileXML(aFileZ,aStrZ);
}



int cAppliMMByPair::Exe()
{
   DoMDT();
   DoCorrel();
   DoBascule();
   DoFusion();
   return 1;
}


int MMByPair_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
   cAppliMMByPair anAppli(argc,argv);


   int aRes = anAppli.Exe();
   BanniereMM3D();
   return aRes;
}





/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
