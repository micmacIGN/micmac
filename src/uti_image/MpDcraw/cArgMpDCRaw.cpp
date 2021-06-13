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
// #include "anag_all.h"

#include "StdAfx.h"

extern const std::string & DefXifOrientation();
namespace NS_MpDcraw
{


cArgMpDCRaw::cArgMpDCRaw(int argc,char ** argv) :
    cAppliBatch
    (
        argc,argv,
    2,   // Trois arguments obligatoires
        1,   // 1 Argument est un fichier
        "MpDCraw"
     ),
     mCons16Bits (1),
     m8BitAdapt  (0),
     mDyn        (0),
     mGammaCorrec      (1.0),
     mEpsLog     (0.3),
     mGB         (0),
     mCB         (0),
     mGR         (0),
     mCR         (0),
     mDiag       (0),
     mConsCol           (0),
     mMastCh     ("R"),
     mScaleMast  (2.0),
     mBayerCalibGeom (0),
     mBicubParam (-0.5),
     mSzSinCard  (3),
     mNameOriIsPrefix (0),
     mAdd16_8B          (1),
     mExpTimeRef        (0),
     mDiaphRef          (0),
     mIsoSpeedRef       (0),
     mExtension         ("MpDcraw"),
     mOfs               (0),
     mDoSplit           (false),
     mSplitMargeInt      (0),
     mSplitMargeExt      (0,0),
     mSwapRB             (-1),
     mUseFF              (1)
{
     LArgMain anArg;
     mNameCLB = "CombLin";

     std::string aInterp="BiLin";

     ElInitArgMain
     (
          ARGC(),ARGV(),
          anArg,
          LArgMain() << EAM(mCons16Bits,"16B",true)
                     << EAM(m8BitAdapt,"8BA",true)
                     << EAM(mGammaCorrec,"Gamma",true)
                     << EAM(mEpsLog,"EpsLog",true)
                     << EAM(mDyn,"Dyn",true)
                 << EAM(mSplit,"Split",true)
                 << EAM(mGB,"GB",true)
                 << EAM(mCB,"CB",true)
                 << EAM(mGR,"GR",true)
                 << EAM(mCR,"CR",true)
                 << EAM(mDiag,"Diag",true)
                 << EAM(mConsCol,"ConsCol",true)
                 << EAM(mClB,"ClB",true)
                 << EAM(mNameCLB,"NameClB",true)
             << EAM(mMastCh,"Master",true)
             << EAM(mScaleMast,"Scale",true)
             << EAM(mCal,"Cal",true)
             << EAM(aInterp,"Interp",true)
             << EAM(mBicubParam,"BiCubParam",true)
             << EAM(mSzSinCard,"SzSinCard",true)
             << EAM(mWB,"WB",true)
             << EAM(mPG,"PG",true)
             << EAM(mNameOriIsPrefix,"Prefix",true)
             << EAM(mAdd16_8B,"Add16B8B",true)
             << EAM(mCamDist,"Dist",true)
             << EAM(mHomolRedr,"HomRedr",true)
             << EAM(mExpTimeRef,"ExpTimeRef",true)
             << EAM(mDiaphRef,"DiaphRef",true)
             << EAM(mIsoSpeedRef,"IsoSpeedRef",true)
             << EAM(mExtension,"Extension",true)
             << EAM(mExtensionAbs,"ExtensionAbs",true)
                     << EAM(mImRef,"ImRef",true)
                     << EAM(mOfs,"Offset",true)
                     << EAM(mDoSplit,"SplitStereo",true)
                     << EAM(mSplitMargeInt,"MargeInt",true)
                     << EAM(mSplitMargeExt,"MargeExt",true)
                     << EAM(mSwapRB,"SwapRB",true)
                     << EAM(mNameOutSpec,"NameOut",true)
                     << EAM(mUseFF,"UseFF",true)
     );



     if (mImRef!="")
     {
         std::string aFulIR = DirChantier() +  mImRef;
         cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aFulIR);
         mExpTimeRef= aMDP.ExpTime() ;
         mDiaphRef=aMDP.Diaph();
         mIsoSpeedRef=aMDP.IsoSpeed();
     }


     if (m8BitAdapt)
        mCons16Bits = 0;


     if (mWB.size()==0)
     {
         mWBSpec = false;
         for (int aK=0 ; aK<3; aK++)
        mWB.push_back(1.0);
     }
     else if (mWB.size()==3)
     {
         mWBSpec = true;
     }
     else
     {
        ELISE_ASSERT(false,"Bad Size for White Balance");
     }

     if (mPG.size()==0)
     {
         mPGSpec = false;
         for (int aK=0 ; aK<3; aK++)
        mPG.push_back(1.0 + (aK==1));
     }
     else if (mPG.size()==3)
     {
         mPGSpec = true;
     }
     else
     {
        ELISE_ASSERT(false,"Bad Size for Gray Pond");
     }






     ELISE_ASSERT
     (
           (mClB.size()==0) ||  (mClB.size()==3) || (mClB.size()==4),
       "Bad size for combin linear"
     );
     if (mClB.size()==3)
     {
        mClB.push_back(mClB.back());
    double aV =    mClB[1]  /2.0;
    mClB[1] = mClB[2] = aV;
     }
     SetNivPurge(eNoPurge);


     if (aInterp=="BiLin")
     {
         mInterp = new cInterpolBilineaire<REAL4>;
     }
     else if (aInterp=="BiCub")
     {
         mInterp = new cInterpolBicubique<REAL4>(mBicubParam);
     }
     else if (aInterp=="SinCard")
     {
         mInterp = new cInterpolSinusCardinal<REAL4>(mSzSinCard);
     }
     else
     {
         ELISE_ASSERT(false,"Unknown Interpolateur");
     }
}

const std::vector<double> &  cArgMpDCRaw::ClB() const
{
   return mClB;
}

bool  cArgMpDCRaw::SwapRB(bool aDef) const
{
  if (mSwapRB==-1) return aDef;

  return (mSwapRB!=0);
}
bool  cArgMpDCRaw::UseFF() const {return mUseFF;}


const std::string &  cArgMpDCRaw::NameCLB() const
{
    return mNameCLB;
}

bool cArgMpDCRaw::IsToSplit(const std::string & aName) const
{
   if (mSplit=="*")
      return true;

  return mSplit.find(aName) != std::string::npos;
}

bool cArgMpDCRaw::GrayBasic() const
{
  return mGB;
}
bool cArgMpDCRaw::ColBasic() const
{
  return mCB;
}

bool cArgMpDCRaw::NameOriIsPrefix() const
{
   return ( mNameOriIsPrefix!=0 );
}

bool cArgMpDCRaw::Diag() const
{
  return mDiag;
}

const bool & cArgMpDCRaw::ConsCol() const
{
  return mConsCol;
}

bool cArgMpDCRaw::GrayReech() const
{
  return mGR;
}
bool cArgMpDCRaw::ColReech() const
{
  return mCR;
}

const std::string & cArgMpDCRaw::Extension() const
{
  return mExtension;
}
const std::string & cArgMpDCRaw::StdExtensionAbs() const
{
  static std::string  ChVide ="";
  return (mExtensionAbs=="None")  ? ChVide : mExtensionAbs;
}

const double & cArgMpDCRaw::Offset() const
{
  return mOfs;
}

int ExtractAngleFromRot(const std::string & aSA,bool & Ok)
{
   Ok = false;
   if (aSA==DefXifOrientation()) return 0;
   static cElRegex anAutomH("Horizontal.*",15);
   if (anAutomH.Match(aSA))
   {
       Ok = true;
       return 0;
   }
   #ifdef REG_EMPTY
       static cElRegex anAutom("Rotate[ ]+(90|180|270)[ ]+(CW.*|.*)",15); // MacOS X does not accept empty (sub)expresions
   #else
       static cElRegex anAutom("Rotate[ ]+(90|180|270)[ ]+(CW|).*",15);
   #endif

   if (! anAutom.Match(aSA)) return 0;


   Ok = true;
   double aVal =  anAutom.VNumKIemeExprPar(1);

   return round_ni(aVal);
}

Fonc_Num GamCor(const cArgMpDCRaw & anArg,Fonc_Num aF,const std::string & aNameFile);



void  cArgMpDCRaw::DevJpg()
{
     std::cout << " MpDcraw with JPEG : use convert (http://doc.ubuntu-fr.org/imagemagick)\n";




     std::string aFullNJPG =  DirChantier()+CurF1();

     std::string aTmp = DirChantier()+ "TmpConvert_XJI_" + StdPrefixGen(CurF1()) + ".tif";

     bool EnGray = GrayBasic() || GrayReech();
     bool En8B = ! Cons16B();

     // double quotes are added to protect spaces in the path
     std::string  aCom =     std::string("\"")+g_externalToolHandler.get( "convert" ).callName()+ "\" "
                         +   aFullNJPG +  std::string(" ")
                         +   std::string(" -compress None ")
                         +   (mConsCol ? " " :   (std::string(" -colorspace ") + (EnGray ? "Gray " : "sRGB ")))
                         +   std::string(" -depth ") + (En8B ? "8 " : "16 ")
                         +   aTmp;

   //std::cout << aCom << "\n";

    VoidSystem(aCom.c_str());

    // Permet de forcer la creation de RGB meme si image en Niv Gris;  inhibe car gere au niveau du ply;
    // a remettre ei autre logiciel a probleme
    if (1 && (! EnGray))
    {
        Tiff_Im aTF(aTmp.c_str());
        int aNbC = aTF.nb_chan();
        if (aNbC != 3)
        {
            std::string aTmp2 = StdPrefix(aTmp+"Gray.tif");
            ELISE_fp::MvFile(aTmp,aTmp2);
            aTF = Tiff_Im(aTmp2.c_str());
            Tiff_Im aTFCol
                    (
                        aTmp.c_str(),
                        aTF.sz(),
                        aTF.type_el(),
                        Tiff_Im::No_Compr,
                        Tiff_Im::RGB
                    );
// �Tiff_Im::Tiff_Im(const char*, Pt2di, Tiff_Im::COMPR_TYPE, GenIm::type_el, Tiff_Im::PH_INTER_TYPE)�
// Tiff_Im::Tiff_Im(const char*, Pt2di, GenIm::type_el, Tiff_Im::COMPR_TYPE, Tiff_Im::PH_INTER_TYPE, L_Arg_Opt_Tiff)

             ELISE_COPY(aTFCol.all_pts(),Virgule(aTF.in(),aTF.in(),aTF.in()),aTFCol.out());
             ELISE_fp::RmFile(aTmp2);
        }
    }


    Tiff_Im aFTmp(aTmp.c_str());
    cMetaDataPhoto aMDP = cMetaDataPhoto::CreateExiv2(aFullNJPG);


   std::string aStrTeta = ICNM()->Assoc1To1("NKS-Assoc-DevlptAutoRot",CurF1(),true);
   int  aITeta;
   FromString(aITeta,aStrTeta);

   if (aITeta!=0)
   {
       std::vector<Im2DGen *>  aV = aFTmp.ReadVecOfIm();
       Im2DGen * aImOut = aV[0]->ImRotate(4-aITeta/90);
       ELISE_fp::RmFile(aTmp);
       Tiff_Im::CreateFromIm(*aImOut,aTmp);
       aFTmp = Tiff_Im(aTmp.c_str());
   }

    if (0) // Apparemment ca cree plus de pb que ca n'en resoud ....
    {
         bool Ok,OkCam;
         int anA = ExtractAngleFromRot( aMDP.Orientation(),Ok);
         int anACam = ExtractAngleFromRot(aMDP.CameraOrientation(),OkCam);

// std::cout << "GGGGG " << anA << " " << anACam << "\n"; getchar();

         if (! OkCam)
         {
             Pt2di aSz = aFTmp.sz();
             // On fait l'hypothese que image prise par un droitier en position standard ,
             // la haut de l'image doit aller a droite
             if (aSz.y > aSz.x)
             {
                 anACam = 270;
                 OkCam = true;
             }
         }

         if (Ok && OkCam)
         {
             if ((anA!=anACam) && (anA==0))
             {
                 int aDA = anACam - anA;
                 if (aDA<0) aDA += 360;
                 // Im2DGen aImIn = aFTmp.ReadIm();
                 std::vector<Im2DGen *>  aV = aFTmp.ReadVecOfIm();
                 Im2DGen * aImOut = aV[0]->ImRotate(4-aDA/90);
// std::cout << "HHHHH " << aV[0]->sz() << aImOut->sz() << "\n";
                 ELISE_fp::RmFile(aTmp);
                 Tiff_Im::CreateFromIm(*aImOut,aTmp);
                 aFTmp = Tiff_Im(aTmp.c_str());
                 // Tiff_Im::CreateFromIm(*aImOut,"toto.tif");
/*
                 Pt2di aSz = aFTmp.sz();

                 std::cout << "ANGLES " << anA << " " << anACam  <<  " " <<   aMDP.Orientation() << "\n";
*/
             }
         }
    }

    std::string aRes = NameRes(CurF1(),"","");
/*
    std::string aRes = DirChantier() +StdPrefixGen( CurF1()) +  + ".tif";

   if (mNameOutSpec!="")
      aRes = mNameOutSpec;
*/




    Tiff_Im aFinal
            (
                aRes.c_str(),
                aFTmp.sz(),
                aFTmp.type_el(),
                aFTmp.mode_compr(),
                aFTmp.phot_interp(),
                ArgMTD()
            );

     Fonc_Num aFRes = aFTmp.in() / FlatField(aMDP,aFullNJPG);

     aFRes = GamCor(*this,aFRes,aFullNJPG);
     if (En8B)
        aFRes = Min(255,aFRes);

     ELISE_COPY
     (
          aFTmp.all_pts(),
          aFRes,
          aFinal.out()
     );

     ELISE_fp::RmFile( aTmp );
}

L_Arg_Opt_Tiff  cArgMpDCRaw::ArgMTD() const
{
   return ArgOpTiffMDP(cMetaDataPhoto::CreateExiv2(DirChantier()+CurF1()),true);
}


std::string cArgMpDCRaw::NameRes(const std::string & aNameFile,const std::string & aName,const std::string & aPref) const
{
   if (mNameOutSpec!="") return mNameOutSpec;

   std::string aN1 = StdPrefix(aNameFile);
   if (mExtensionAbs !="")
   {
       return DirChantier() +aN1+StdExtensionAbs()+".tif";
   }

   std::string aN2 =  Extension();

   if (Add16_8B())
       aN2 = aN2+ (Cons16B()? "16B" : "8B") ;

   if ( NameOriIsPrefix())
      ElSwap(aN1,aN2);

   return  DirChantier()
         + aPref
         + aN1
         + std::string("_")
         + aN2
         + std::string("_")
         + aName
         +  std::string(".tif");

}

bool IsKnownPngPost(const std::string & aPost)
{
    return    (aPost == "png")
           || (aPost == "PNG");
}


void cArgMpDCRaw::Exec()
{
   if (IsPostfixed(CurF1()))
   {
      std::string aPost = StdPostfix(CurF1());
       if (IsKnownJPGPost(aPost) || IsKnownPngPost(aPost))
       {
            DevJpg();
            return;
       }
   }
   // En mode raw , on est jrs en coul ?
   if (mConsCol)
   {
       mGB = 0;
       mCB = 1;
   }

   std::string aCal = mCal;
   if (aCal!="")
   {
         std::string aCal0 = aCal;
         if (!  ELISE_fp::exist_file(aCal))
            aCal = DirChantier()+aCal;


         if (!  ELISE_fp::exist_file(aCal))
            aCal = DirChantier()+ICNM()->Assoc1To1(aCal0,CurF1(),true);

         if (!  ELISE_fp::exist_file(aCal))
         {
             std::cout << "For KEY = " << aCal0 << "\n";
             std::cout << "For name = " << aCal << "\n";
             ELISE_ASSERT
             (
                 false,
                 "Cannot get calib file"
             );
         }

         delete mBayerCalibGeom;
         mBayerCalibGeom = new cBayerCalibGeom
                           (
                      StdGetObjFromFile<cBayerCalibGeom>
                  (
                                      aCal,
                      StdGetFileXMLSpec("SuperposImage.xml"),
                      "BayerCalibGeom",
                      "BayerCalibGeom"
                  )
                   );
   }
   cNChannel aNC = cNChannel::Std(*this,DirChantier()+CurF1());

   aNC.SauvInit();
}

double cArgMpDCRaw::Dyn() const
{
   return mDyn;
}

double cArgMpDCRaw::Gamma( const std::string& aNameIm) const
{
   if (! EAMIsInit(const_cast<void *>((void *)&mGammaCorrec)))
   {
         if (!  mCons16Bits)
         {
              std::string aNameGama = ICNM()->Assoc1To1("NKS-Assoc-STD-Gama8Bits",aNameIm,true);
              double aRes;
              FromString(aRes,aNameGama);
              return aRes;
         }
   }

    return mGammaCorrec;
}


double cArgMpDCRaw::EpsLog() const { return mEpsLog; }

bool cArgMpDCRaw::Cons16B() const
{
   return ( mCons16Bits!=0 );
}

bool cArgMpDCRaw::Adapt8B() const
{
   return ( m8BitAdapt!=0 );
}


const std::string &  cArgMpDCRaw::MastCh() const
{
   return mMastCh;
}


const double &   cArgMpDCRaw::ScaleMast() const { return mScaleMast; }
const double &   cArgMpDCRaw::ExpTimeRef() const { return mExpTimeRef; }
const double &   cArgMpDCRaw::DiaphRef() const { return mDiaphRef; }
const double &   cArgMpDCRaw::IsoSpeedRef() const { return mIsoSpeedRef; }


cBayerCalibGeom *   cArgMpDCRaw::BayerCalib() const
{
   return mBayerCalibGeom;
}


cInterpolateurIm2D<REAL4> * cArgMpDCRaw::Interpol() const
{
   return mInterp;
}


const std::vector<double>  &  cArgMpDCRaw::WB() const
{
    return mWB;
}
bool  cArgMpDCRaw::WBSpec () const
{
  return mWBSpec;
}


const std::vector<double>  &  cArgMpDCRaw::PG() const
{
    return mPG;
}
bool  cArgMpDCRaw::PGSpec () const
{
  return mPGSpec;
}

bool cArgMpDCRaw::Add16_8B() const
{
   return ( mAdd16_8B!=0 );
}

const std::string &  cArgMpDCRaw::CamDist() const
{
    return mCamDist;
}

const std::string &  cArgMpDCRaw::HomolRedr() const
{
    return mHomolRedr;
}

const bool  & cArgMpDCRaw::DoSplit() const {return mDoSplit;}
const int   & cArgMpDCRaw::SpliMargeInt() const {return mSplitMargeInt;}
const Pt2di & cArgMpDCRaw::SpliMargeExt() const {return mSplitMargeExt;}


};



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
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
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
