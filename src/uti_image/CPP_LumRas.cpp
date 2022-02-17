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



template <class Type> class cSomValCC :  public cCC_NoActionOnNewPt
{
    public :
      cSomValCC(Type & aIm) :
         mIm(aIm) ,
         mSom (0)
      {
      }
      void  OnNewPt(const Pt2di &aPt) { mSom += mIm.get(aPt);}

      Type & mIm;
      double mSom;
};

template <class Type> class cMarqImCC : public cCC_NoActionOnNewPt
{
    public :
      cMarqImCC(Type & aIm,int aVal) :
         mIm(aIm) ,
         mVal (aVal)
      {
      }
      void  OnNewPt(const Pt2di &aPt) {  mIm.oset(aPt,mVal);}



      Type & mIm;
      double    mVal;
};




//   aImIn peut etre egale a ImOut
//

template  <class TypeEtiq,class TypeImIn,class TypeImOut>
          void    MoyByCC(bool V4,TypeEtiq  aTIm,int aValExclu,TypeImIn  aImIn,TypeImOut  aImOut)
{
   Pt2di aSz = aTIm.sz();

   Im2D_Bits<1> aMasq1 = ImMarqueurCC(aSz);
   TIm2DBits<1> aTMasq1(aMasq1);

   Im2D_Bits<1> aMasq2(aSz.x,aSz.y,1);
   TIm2DBits<1> aTMasq2(aMasq2);
   ELISE_COPY(aMasq2.border(1),0,aMasq2.out());

   Pt2di aP;
   for(aP.x=0 ; aP.x<aSz.x ; aP.x++)
   {
      for(aP.y=0 ; aP.y<aSz.y ; aP.y++)
      {
          int aValIm = aTIm.get(aP);
          if ((aValIm!=aValExclu) && (aTMasq1.get(aP)==1))
          {
               cSomValCC<TypeImIn> aCumSom(aImIn);
               int aNb = OneZC(aP,V4,aTMasq1,1,0,aTIm,aValIm,aCumSom);

               cMarqImCC<TypeImOut> aMarqIm(aImOut,aCumSom.mSom/aNb);
               OneZC(aP,V4,aTMasq2,1,0,aTIm,aValIm,aMarqIm);
          }
      }
   }
}


class cImage_LumRas;
class cAppli_LumRas;

class cImage_LumRas
{
    public :
       friend class cAppli_LumRas;
       cImage_LumRas(const std::string& aName,cAppli_LumRas & anAppli);

       std::string      mNameFull;
       std::string      mDir;
       std::string      mName;
       cAppli_LumRas &  mAppli;
       Tiff_Im          mTiffIm;
       Im2D_REAL4       mImShade;
       Im2D_U_INT2      mIm;
        

       Fonc_Num         FMoy(int aKIter,int aSzW,Fonc_Num);
       Fonc_Num         FLoc(int aKIter,int aSzW,Im2D_U_INT2);
       Fonc_Num         MoyGlobImage(Fonc_Num aF);
       // Fonc_Num         MoyByCC(Fonc_Num aF);
       void CalculShadeByDiff(int aNbIter,int aSzW);
};

class cAppli_LumRas : cAppliWithSetImage
{
    public :
       friend class cImage_LumRas;
       cAppli_LumRas(int argc,char ** argv);
    private :
       void  DoShadeByLeastSquare();

       std::string mNameImBase;
       //Tiff_Im *   mTifBaseGr;
       Tiff_Im *   mTifBaseCoul;
       std::string mPatImRas;
       std::string mPostMasq;
       std::vector<cImage_LumRas *> mVIm;
       std::string                  mKeyHom;
       Im2D_U_INT2                  mImGr;
       Pt2di                        mSz;
       Im2D_Bits<1>                 mImMasq;
       std::string                  mNameTargSh;
       int                          mSzW;
       int                          mNbIter;

};

/**********************************************************************/
/*                                                                    */
/*                  cImage_LumRas                                     */
/*                                                                    */
/**********************************************************************/


Fonc_Num   cImage_LumRas::FMoy(int aNbIter,int aSzW,Fonc_Num aF)
{
   Fonc_Num aRes = Rconv(aF);
   for (int aK=0 ; aK<aNbIter; aK++)
      aRes = rect_som(aRes,aSzW) / ElSquare(1.0+2*aSzW);

   return aRes;
}

Fonc_Num  cImage_LumRas::MoyGlobImage(Fonc_Num aF)
{
   Im2D_Bits<1> aIM = mAppli.mImMasq;
   Fonc_Num aFMasq = aIM.in(0);
   double aVS[2];

   ELISE_COPY
   (
        aIM.all_pts(),
        Virgule(aF*aFMasq,aFMasq),
        sigma(aVS,2)
   );

   return aVS[0] / aVS[1];
}


Fonc_Num     cImage_LumRas::FLoc(int aNbIter,int aSzW,Im2D_U_INT2 anIm)
{
   Fonc_Num aFMasq = mAppli.mImMasq.in(0);
   Fonc_Num aF = anIm.in(0);


   Fonc_Num aFMoy =  0;

   if (aNbIter>0)
      aFMoy =  FMoy(aNbIter,aSzW,aF*aFMasq) / Max(1e-2,FMoy(aNbIter,aSzW,aFMasq)) ;
   else if (0)
      aFMoy = MoyGlobImage(aF);
   else if (1)
   {
        Pt2di aSz = anIm.sz();
        Im2D_REAL4 aIMoy (aSz.x,aSz.y);
        ::MoyByCC(true, TIm2DBits<1>(mAppli.mImMasq),0,TIm2D<U_INT2,INT>(anIm),TIm2D<REAL4,REAL8>(aIMoy));
        aFMoy = aIMoy.in();

Tiff_Im::Create8BFromFonc("TestMasq.tif",aSz,mAppli.mImMasq.in()*255);
        //Tiff_Im::Create8BFromFonc("Test.tif",aSz,aFMoy); std::cout << "CCCCCCCCCC\n"; getchar();
   }

   return  (aF / Max(1e-2,aFMoy)) * aFMasq;
}

cImage_LumRas::cImage_LumRas(const std::string& aNameFull,cAppli_LumRas & anAppli) :
   mNameFull   (aNameFull),
   mAppli      (anAppli),
   mTiffIm     (Tiff_Im::StdConvGen(aNameFull,1,true)),
   mImShade    (1,1),
   mIm         (1,1)
{
   SplitDirAndFile(mDir,mName,mNameFull);

   mIm = Im2D_U_INT2::FromFileStd(mNameFull);
   if (mAppli.mKeyHom !="")
   {
       Pt2di aSzIn = mIm.sz();
       std::string aNameH = mAppli.mEASF.mICNM->Assoc1To2(mAppli.mKeyHom,mName,NameWithoutDir(mAppli.mNameImBase),true);
       std::cout << "SZ IM " << aSzIn << " " << mNameFull << " " << aNameH << "\n";

       ElPackHomologue aPack = ElPackHomologue::FromFile(aNameH);
       cElHomographie aHom = cElHomographie::RansacInitH(aPack,1000,1000);
       aHom = aHom.Inverse();

       Pt2di aSz = mAppli.mSz;
       Im2D_U_INT2 anImReech(aSz.x,aSz.y);
       TIm2D<U_INT2,INT> aTR(anImReech);
       TIm2D<U_INT2,INT> aT0(mIm);
       Pt2di aP;

       for (aP.x=0 ; aP.x<aSz.x ; aP.x++)
       {
           for (aP.y=0 ; aP.y<aSz.y ; aP.y++)
           {
                aTR.oset(aP,aT0.getr(aHom.Direct(Pt2dr(aP)),0));
           }
       }


       mIm = anImReech;
   }
}

void cImage_LumRas::CalculShadeByDiff(int aNbIter,int aSzW)
{

   mImShade.Resize(mAppli.mImGr.sz());

   std::string aNameOut = mDir+ "LumRas_"+StdPrefix(mName) + ".tif";
   Tiff_Im TifTest
           (
                 aNameOut.c_str(),
                 mIm.sz(),
                 // GenIm::u_int1,
                 GenIm::real4,
                 Tiff_Im::No_Compr,
                 Tiff_Im::BlackIsZero
           );

    Fonc_Num aFRas  =  FLoc(aNbIter,aSzW,mIm);
    Fonc_Num aFStd  =  FLoc(aNbIter,aSzW,mAppli.mImGr);
    Tiff_Im::Create8BFromFonc("Test-Ras.tif",mIm.sz(),aFRas*100);
    Tiff_Im::Create8BFromFonc("Test-Std.tif",mIm.sz(),aFStd*100);
// Fonc_Num     cImage_LumRas::FLoc(int aNbIter,int aSzW,Fonc_Num aF)

   ELISE_COPY(mImShade.all_pts(),(aFRas-aFStd),mImShade.out());

   ELISE_COPY
   (
      TifTest.all_pts(),
      // Max(0,Min(255,128 * (1 + 2*mImShade.in()))),
      mImShade.in(),
      TifTest.out()
   );

}



/**********************************************************************/
/*                                                                    */
/*                  cAppli_LumRas                                     */
/*                                                                    */
/**********************************************************************/

void  cAppli_LumRas::DoShadeByLeastSquare()
{

   Im2D_Bits<1> aMarqueur = ImMarqueurCC(mSz);
   TIm2DBits<1> aTMarqueur(aMarqueur);
   TIm2DBits<1> aTMasq(mImMasq);

   Pt2di aP;
   for (aP.x =0 ; aP.x<mSz.x  ; aP.x++)
   {
       for (aP.y =0 ; aP.x<mSz.y  ; aP.y++)
       {
            if (aTMasq.get(aP) && aTMarqueur.get(aP))
            {
            }
       }
   }

   
}



cAppli_LumRas::cAppli_LumRas(int argc,char ** argv) :
   cAppliWithSetImage(argc-2,argv +2,TheFlagNoOri|TheFlagDev16BGray),
   // mTifBaseGr   (0),
   mTifBaseCoul (0),
   mImGr        (1,1),
   mImMasq      (1,1),
   mSzW        (-1),
   mNbIter     (-1)

{
     std::vector<double> aPdsI;
     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAM(mNameImBase, "Image name",  true, "Image name", eSAM_IsExistFile)
                      << EAM(mPatImRas, "Image pattern", true, "Pattern", eSAM_IsPatFile) ,
           LArgMain() << EAM(mPostMasq,"Masq",true,"Mask for computation", eSAM_NoInit)
                      << EAM(aPdsI,"PdsIn",true,"Pds on RGB Input, def=[1,1,1]", eSAM_NoInit)
                      << EAM(mNameTargSh,"TargShade",true,"Targeted Shade", eSAM_NoInit)
                      << EAM(mSzW,"SzW",true,"Sz of average window", eSAM_NoInit)
                      << EAM(mNbIter,"NbIter",true,"Number of window iter", eSAM_NoInit)
    );


    if (!MMVisualMode)
    {
        for (int aK=(int)aPdsI.size() ; aK<3 ; aK++)
            aPdsI.push_back(1);
        // mTifBaseGr =   new  Tiff_Im (Tiff_Im::StdConvGen(mNameImBase,1,true));
        mTifBaseCoul = new  Tiff_Im (Tiff_Im::StdConvGen(mNameImBase,3,true));

        mSz =  mTifBaseCoul->sz();
        mImGr.Resize(mSz);
        Symb_FNum aFCoul(mTifBaseCoul->in());
        Fonc_Num aFGr =  (aPdsI[0]*aFCoul.v0()+aPdsI[1]*aFCoul.v1()+aPdsI[2]*aFCoul.v2())/(aPdsI[0]+aPdsI[1]+aPdsI[2]);

        ELISE_COPY(mImGr.all_pts(),aFGr,mImGr.out());


        mImMasq = Im2D_Bits<1>(mSz.x,mSz.y,1);
        if (EAMIsInit(&mPostMasq))
        {
            CorrecNameMasq(mEASF.mDir,NameWithoutDir(mNameImBase),mPostMasq);
            std::string aNameMasq = StdPrefix(mNameImBase)+mPostMasq+".tif";
            Tiff_Im aTM(aNameMasq.c_str());
            ELISE_COPY(mImMasq.all_pts(),aTM.in(0),mImMasq.out());
        }
        ELISE_COPY(mImMasq.border(1),0,mImMasq.out());

        mKeyHom = "NKS-Assoc-CplIm2Hom@@dat";
        // mKeyHom = "";

        Fonc_Num aGlobSh;
        for (int aK=0 ; aK<int(mVSoms.size()) ; aK++)
        {
            std::string aName = mVSoms[aK]->attr().mIma->mNameIm;
            mVIm.push_back(new cImage_LumRas(mEASF.mDir+aName,*this));
            //Fonc_Num aFShade = mVIm.back()->mImShade.in();



            // aGlobSh = (aK==0) ? aFShade : Virgule(aGlobSh,aFShade);
        }


        if (EAMIsInit(&mNameTargSh))
        {
            DoShadeByLeastSquare();
        }
        else
        {
             for (int aK=0 ; aK<int(mVIm.size()) ; aK++)
                mVIm[aK]->CalculShadeByDiff(mNbIter,mSzW);
        }
/* 
  // RGB 

       std::string aNameOut = mEASF.mDir+ "LumRas_"+StdPrefix(mNameImBase) + ".tif";
       Tiff_Im TifTest
               (
                     aNameOut.c_str(),
                     mSz,
                     GenIm::u_int1,
                     Tiff_Im::No_Compr,
                     Tiff_Im::RGB
               );


       ELISE_COPY
       (
             TifTest.all_pts(),
             // Max(0,Min(255,128 * (1 + 5*aGlobSh))),
             Max(0,Min(255,aFCoul+ 20*aGlobSh)),
             TifTest.out()
       );
*/
    }
}


int LumRas_main(int argc,char ** argv)
{
     cAppli_LumRas anALR(argc,argv);


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
