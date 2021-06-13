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


class cAppli_AddNoiseImage
{
   public :
     std::string mNameIn;
     std::string mNameOut;
     double      mNoise; 
     double      mPivot; // En cas de chgt de dyn
     Pt2di       mSz;
     bool        mResFloat;
     bool        mInitRan;

cAppli_AddNoiseImage(int argc,char **argv) :
    mPivot (128),
    mResFloat (false),
    mInitRan  (false)
{
   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAMC(mNameIn, "Image", eSAM_IsExistFile)
                   << EAMC(mNoise, "Basic uncorrelated noise"),
       LArgMain()  << EAM(mNameOut,"Out",true)
                   << EAM(mPivot,"Pivot",true,"Pivot level when chang dyn")
                   << EAM(mResFloat,"Float",true,"If true generate float image (def maintain type)")
                   << EAM(mInitRan,"InitRan",true,"If true initialize random generator with time  def=false ")
   );

   if (mInitRan)
      NRrandom3InitOfTime();

   Tiff_Im aTifIn = Tiff_Im::StdConvGen(mNameIn,-1,true);
   mSz = aTifIn.sz();

   if (! EAMIsInit(&mNameOut))
      mNameOut = "Noised-" + StdPrefix(mNameIn) + ".tif";


   GenIm::type_el aTypeEl = mResFloat ? GenIm::real4 : aTifIn.type_el() ;
   
   Tiff_Im aTifOut
           ( 
              mNameOut.c_str(),
              mSz,
              aTypeEl,
              Tiff_Im::No_Compr,
              aTifIn.phot_interp()
           );

   Fonc_Num aRes = aTifIn.in();
   aRes = aRes +  mNoise *frandr();
   aRes = Tronque(aTypeEl,aRes);

   ELISE_COPY
   (
        aTifOut.all_pts(),
        aRes,
        aTifOut.out()
   );

};

};

int CPP_AddNoiseImage(int argc,char ** argv)
{
   cAppli_AddNoiseImage anAppli(argc,argv);

   return EXIT_SUCCESS;
}


/***************************************************************/

class cAppli_SimulDep
{
    public :

        cAppli_SimulDep(int argc,char ** argv);

    private :
         void OneMatch(const std::string & anInterp,int aK);
         void MakeMedian(const std::string & anInterp,bool X,int aKITer);
         void GenerateNoise(const std::string & anImIn,const std::string & anImOut);

         std::string DirBuf(const std::string & anInterp) const {return   "Accum-" + anInterp +"/";}
         std::string FileDepl(const std::string & anInterp,int aK,bool X) const 
         {
             return   DirBuf(anInterp) + std::string(X ? "DeplX-" : "DeplY-") +  ToString(aK) + ".tif";
         }

         std::string   mIm1;
         std::string   mIm2;
         std::string   mNoiseIm1;
         std::string   mNoiseIm2;
         int           mLevelRand;
         int           mNbTir;
         std::string   mParamMM;
         bool          mDoMatch;
         int           mPerRes;
};

void cAppli_SimulDep::GenerateNoise(const std::string & anImIn,const std::string & anImOut)
{
    std::string  aCom = MM3dBinFile_quotes("SimuLib AddNoise ")  
                        + " " + anImIn
                        + " " + ToString(mLevelRand) 
                        + std::string(" InitRan=true")
                        + " Out=" + anImOut;
   System(aCom);
}

void cAppli_SimulDep::MakeMedian(const std::string & anInterp,bool X,int aKIter)
{
   std::vector<Im2D_REAL4> aVIm;
   for (int aK=0 ; aK<= aKIter ; aK++)
   {
       std::string aNameDepl = FileDepl(anInterp,aK,X);
       // Tiff_Im aTif(aNameDepl.c_str());
       aVIm.push_back(Im2D_REAL4::FromFileStd(aNameDepl));
   }
   Pt2di aSz = aVIm[0].sz();
   Im2D_REAL4 aMed(aSz.x,aSz.y);
   Im2D_REAL4 aMoy(aSz.x,aSz.y);
   Im2D_REAL4 aMoyP(aSz.x,aSz.y);

   Pt2di aP;
   for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
   {
       for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
       {
           std::vector<double> aVV;
           double aSomV=0;
           for (int aK=0 ; aK<= aKIter ; aK++)
           {
               double aVal = aVIm[aK].GetR(aP);
               aSomV += aVal;
               aVV.push_back(aVal);
           }
           aMed.SetR(aP,MedianeSup(aVV));
           aMoy.SetR(aP,aSomV/(1+aKIter));
           std::sort(aVV.begin(),aVV.end());
           double aSomVP=0;
           double aSomP=0;
           for (int aK=0 ; aK<=aKIter ; aK++)
           {
               double aVal = aVIm[aK].GetR(aP);
               double aP = ElMin(aK+1,aKIter+1-aK);
               aSomVP += aVal * aP;
               aSomP  += aP;
           }
           aMoyP.SetR(aP, aSomVP/aSomP);
       }
   }
   Tiff_Im::CreateFromIm(aMed,DirBuf(anInterp)+"Med"+ anInterp+  std::string(X?"X":"Y") + ".tif");
   Tiff_Im::CreateFromIm(aMoy,DirBuf(anInterp)+"Moy"+ anInterp+  std::string(X?"X":"Y") + ".tif");
   Tiff_Im::CreateFromIm(aMoyP,DirBuf(anInterp)+"PMoy"+ anInterp+  std::string(X?"X":"Y") + ".tif");
}

void cAppli_SimulDep::OneMatch(const std::string & anInterp,int aK)
{
    GenerateNoise(mIm1,mNoiseIm1);
    GenerateNoise(mIm2,mNoiseIm2);
    ELISE_fp::PurgeDir("Pyram/");
    std::string aDirMEC = "MM-" + anInterp + "/";
    std::string  aCom = MM3dBinFile_quotes("MICMAC") 
                           + " "      + mParamMM
                           + " +Im1=" + mNoiseIm1
                           + " +Im2=" + mNoiseIm2
                           + std::string(" +DirMEC=") + aDirMEC
                           + std::string(" +Interp=eInterpol") + anInterp;
                        ;

   System(aCom);
   std::string aDirBuf = DirBuf(anInterp);
   ELISE_fp::MkDirSvp(aDirBuf);
   ELISE_fp::CpFile(aDirMEC + "Px1_Num9_DeZoom1_LeChantier.tif", FileDepl(anInterp,aK,true));
   ELISE_fp::CpFile(aDirMEC + "Px2_Num9_DeZoom1_LeChantier.tif", FileDepl(anInterp,aK,false));

}

cAppli_SimulDep::cAppli_SimulDep(int argc,char ** argv) :
    mNbTir   (25),
    mDoMatch (true),
    mPerRes  (5)
{
   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAMC(mIm1    , "Image 1", eSAM_IsExistFile)
                   << EAMC(mIm2    , "Image 2", eSAM_IsExistFile)
                   << EAMC(mParamMM, "Param MM", eSAM_IsExistFile)
                   << EAMC(mLevelRand, "NivRand", eSAM_IsExistFile)
       ,
       LArgMain()  << EAM(mNbTir  ,"NbRand" ,true,"Number of random iter")
                   << EAM(mDoMatch,"DoMatch",true,"Do also the matching")
   );
   mNoiseIm1 = "Noised-"+ mIm1;
   mNoiseIm2 = "Noised-"+ mIm2;


   if (mDoMatch)
   {
      for (int aK=0 ; aK<mNbTir ; aK++)
      {
          OneMatch("BiCub",aK);
          OneMatch("BiLin",aK);
          OneMatch("SinCard",aK);
          OneMatch("MPD",aK);

          if ( ((aK+1)==mNbTir) || ( ((aK+1)%mPerRes) ==0)  )
          {
               MakeMedian("SinCard",true,aK);
               MakeMedian("SinCard",false,aK);
               MakeMedian("BiLin",true,aK);
               MakeMedian("BiLin",false,aK);
               MakeMedian("MPD",true,aK);
               MakeMedian("MPD",false,aK);
               MakeMedian("BiCub",true,aK);
               MakeMedian("BiCub",false,aK);
          }
      }
   }



}

int CPP_SimulDep(int argc,char ** argv)
{
    cAppli_SimulDep anAppli(argc,argv);

    return EXIT_SUCCESS;
}

/***************************************************************/

class cProfilImage
{
    public :
       cProfilImage(int argc,char ** argv);

       std::string   mNameIm;
       bool          mProfX;
       std::string   mNameMasq;
       std::string   mPrefix;
       int           mSz;
       Pt2di         mSzIm;
       Fonc_Num      mFPond;
       Im1D_REAL8    mImVals;
       Im1D_REAL8    mImPds;
       Im1D_REAL8    mImAbsc;
       double        mSeuilPds;
};

cProfilImage::cProfilImage(int argc,char ** argv) :
    mFPond    (1.0),
    mImVals   (1),
    mImPds    (1),
    mImAbsc   (1),
    mSeuilPds (0.0)
{
   ElInitArgMain
   (
       argc,argv,
       LArgMain()  << EAMC(mNameIm  ,"Image", eSAM_IsExistFile)
                   << EAMC(mProfX, "Profil in X or Y", eSAM_IsBool),
       LArgMain()  << EAM(mNameMasq,"NbRand",true,"Number of random iter")
                   << EAM(mPrefix,"Prefix",true,"Prefix to generate output")
   );

   if (!EAMIsInit(&mPrefix))
       mPrefix = "Profil-" +  std::string(mProfX ? "X" : "Y") + "-" +StdPrefix(mNameIm);

   Tiff_Im aTifIm = Tiff_Im::StdConvGen(mNameIm,1,true);
   mSzIm = aTifIm.sz();

   mSz = mProfX ? mSzIm.x : mSzIm.y;
   Fonc_Num fProj = mProfX ? FX : FY;

   mImVals = Im1D_REAL8(mSz,0.0);
   mImPds =  Im1D_REAL8(mSz,0.0);
   mImAbsc = Im1D_REAL8(mSz,0.0);

   ELISE_COPY
   (
      aTifIm.all_pts(),
      Virgule(aTifIm.in()*mFPond,fProj*mFPond,mFPond),
      Virgule(mImVals.histo(),mImAbsc.histo(),mImPds.histo()).chc(fProj)
   );

   double * aDV = mImVals.data();
   double * aDP = mImPds.data();
   double * aDA = mImAbsc.data();

   FILE * aFPabsc = FopenNN(mPrefix+"-absc.txt","w","cProfilImage");
   FILE * aFPord  = FopenNN(mPrefix+"-ord.txt","w","cProfilImage");

   for (int aK=0 ; aK<mSz ; aK++)
   {
       double aPds = aDP[aK];
       if (aPds>mSeuilPds)
       {
           fprintf(aFPabsc,"%lf\n",aDA[aK]/aPds);
           fprintf(aFPord,"%lf\n",aDV[aK]/aPds);
       }
   }
   fclose(aFPabsc);
   fclose(aFPord);
}




int CPP_ProfilImage(int argc,char ** argv)
{
   cProfilImage anAppli(argc,argv);

   return EXIT_SUCCESS;
}





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
