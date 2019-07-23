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


class cAppli_CoronaRessample
{
   public :
      cAppli_CoronaRessample(int argc,char ** argv);
   private :
      std::string          mImIn;
      std::vector<double>  mDxOfX;
      std::vector<double>  mDxOfY;
      std::vector<double>  mDyOfX;
      std::vector<double>  mDyOfY;

      std::vector<double>  mParamDxOfX;
      std::vector<double>  mParamDxOfY;
      std::vector<double>  mParamDyOfX;
      std::vector<double>  mParamDyOfY;
      std::string          mOut;

      void InitParam(std::vector<double> & aVRes,const std::vector<double> & aParam,int aSz);

};
void cAppli_CoronaRessample::InitParam(std::vector<double> & aVRes,const std::vector<double> & aParam,int aSz)
{
    aVRes = std::vector<double>(aSz,0);
    if (aParam.size()==0)
       return;
    ELISE_ASSERT(aParam.size()==4,"CoronaRessampl bad number of param");

    double aX0 = aParam.at(0);
    double aX1 = aParam.at(1);
    int   aNb = aParam.at(2);
    double aAmpl = aParam.at(3);
    double aPer = (aX1-aX0) / double(aNb);
    for (int aK=0 ; aK<aSz ; aK++)
    {
         double aPhase = (aK-aX0) / aPer + 0.5 ;  // we want extrem values at limits
         aPhase = aPhase -round_ni(aPhase);
         aVRes.at(aK) = aPhase * aAmpl;
    }
}

typedef U_INT2 TypeEl;
typedef INT    TypeBase;
typedef Im2D<TypeEl,TypeBase> TypeIm;

cAppli_CoronaRessample::cAppli_CoronaRessample(int argc,char ** argv)
{
     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(mImIn,"Input Image", eSAM_IsPatFile)
                      << EAMC(mParamDxOfX,"Param dx, function of X,[X0,X1,Nb,Step] or []", eSAM_IsExistFile)
                      << EAMC(mParamDxOfY,"Param dx, function of Y,[Y0,Y1,Nb,Step] or []", eSAM_IsExistFile)
                      << EAMC(mParamDyOfX,"Param dy, function of X,[X0,X1,Nb,Step] or []", eSAM_IsExistFile)
                      << EAMC(mParamDyOfY,"Param dy, function of Y,[Y0,Y1,Nb,Step] or []", eSAM_IsExistFile),
           LArgMain() << EAM(mOut,"Out",true,"Name for result")
     );
     Tiff_Im  aTF = Tiff_Im::StdConvGen(mImIn,1,true);
     if (! EAMIsInit(&mOut))
     {
        mOut = DirOfFile(mImIn) + "CoRessample-"+ NameWithoutDir(mImIn) ;
     }
     Tiff_Im  aTOut
              (
                     mOut.c_str(),
                     aTF.sz(),
                     aTF.type_el(),
                     Tiff_Im::No_Compr,
                     Tiff_Im::BlackIsZero
              );

     Pt2di aSz = aTF.sz();
     std::cout << "Szzzz " << aSz << "\n";

     InitParam(mDxOfX,mParamDxOfX,aSz.x);
     InitParam(mDxOfY,mParamDxOfY,aSz.y);
     InitParam(mDyOfX,mParamDyOfX,aSz.x);
     InitParam(mDyOfY,mParamDyOfY,aSz.y);

     double aSzK = 5.0; // anEt.SzSinCard().ValWithDef(5.0);
     double aSzA =5.0; //  anEt.SzAppodSinCard().ValWithDef(5.0);
     int aNbD = 1000; //  anEt.NdDiscKerInterp().ValWithDef(1000);

     cSinCardApodInterpol1D aKer(cSinCardApodInterpol1D::eTukeyApod,aSzK,aSzA,1e-4,false);
     cInterpolateurIm2D<TypeEl>  * anInterp = new cTabIM2D_FromIm2D<TypeEl>(&aKer,aNbD,false);

     TypeIm aInput = TypeIm::FromFileStd(mImIn);
     TypeIm aOutPut(aSz.x,aSz.y,TypeBase(0));

     Pt2di aPOut;
     int aRab=0;
     for (aPOut.x = aRab; aPOut.x < aSz.x - aRab ; aPOut.x++)
     {
         if  (((aPOut.x+1) %100)==0)
            std::cout << "Remain " <<  (aSz.x-aPOut.x ) << "\n";
         for (aPOut.y = aRab; aPOut.y < aSz.y - aRab ; aPOut.y++)
         {
             // Pt2dr aPIn(
             Pt2dr  aDx(mDxOfX.at(aPOut.x)+mDxOfY.at(aPOut.y),0);
             Pt2dr  aDy(0,mDyOfX.at(aPOut.x)+mDyOfY.at(aPOut.y));

             Pt2dr aPIn = Pt2dr(aPOut) + aDx + aDy;
             double aVal = aInput.Get(aPIn,*anInterp,0.0);
             aOutPut.SetI(aPOut,round_ni(aVal));
         }
     }
     ELISE_COPY(aOutPut.all_pts(),aOutPut.in(),aTOut.out());
}

int CoronaRessample_main(int argc,char ** argv)
{
     MMD_InitArgcArgv(argc,argv);
     cAppli_CoronaRessample  anAppi(argc,argv);

    return EXIT_SUCCESS;
}




int Undist_main(int argc,char ** argv)
{
     MMD_InitArgcArgv(argc,argv);


     std::string aFullName;
     std::string aKeyOri;

     const char * aStrIC = "XX@ABB_InternalCall";
     bool aCanUseMkF=false;
     bool aIC=false;

     ElInitArgMain
     (
           argc,argv,
           LArgMain() << EAMC(aFullName,"Image name (Dir + Pat)", eSAM_IsPatFile)
                      << EAMC(aKeyOri,"Orientation (Calibration or Orientation File or Directory of Orientation)", eSAM_IsExistFile),
           LArgMain() << EAM(aCanUseMkF,"UseMkF",true,"Can use make file for parallelization, def=true")
              << EAM(aIC,aStrIC,true,"Internal use", eSAM_InternalUse)
    );

    std::string aDir,aPat;
    SplitDirAndFile(aDir,aPat,aFullName);
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);


    const cInterfChantierNameManipulateur::tSet * aSetIm = anICNM->Get(aPat);

    if ( (!aIC) && aCanUseMkF && (aSetIm->size() >1))
    {
       return 0;
    }

    for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
    {
         std::string aNameIm = (*aSetIm)[aKIm];
         std::string aNameCal = aDir+aKeyOri;
         if (!ELISE_fp::exist_file(aNameCal))
             aNameCal = anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aKeyOri,aNameIm,true);
         CamStenope * aCam = CamOrientGenFromFile(aNameCal,anICNM);
         Pt2di aSz = aCam->Sz();
         std::cout << aCam->Focale() << aSz << " " << aNameIm << "\n";



    }
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
