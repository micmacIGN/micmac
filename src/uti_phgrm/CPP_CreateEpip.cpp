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



cCpleEpip * StdCpleEpip
          (
             std::string  aDir,
             std::string  aNameOri,
             std::string  aNameIm1,
             std::string  aNameIm2
          )
{
    if (aNameIm1 > aNameIm2) ElSwap(aNameIm1,aNameIm2);
    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::string aNameCam1 =  anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aNameOri,aNameIm1,true);
    std::string aNameCam2 =  anICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aNameOri,aNameIm2,true);

    CamStenope * aCam1 = CamStenope::StdCamFromFile(true,aNameCam1,anICNM);
    CamStenope * aCam2 = CamStenope::StdCamFromFile(true,aNameCam2,anICNM);
    return new cCpleEpip (aDir,1,*aCam1,aNameIm1,*aCam2,aNameIm2);

}

class cApply_CreateEpip_main
{
   public :
      cApply_CreateEpip_main(int argc,char ** argv);

     cInterfChantierNameManipulateur * mICNM;


      int mDegre;
      double mLengthMax ;
      cBasicGeomCap3D *  mGenI1;
      cBasicGeomCap3D *  mGenI2;

      void DoEpipGen();

      Pt2dr DirEpipIm2(cBasicGeomCap3D * aG1,cBasicGeomCap3D * aG2,int aNb,ElPackHomologue & aPack,bool AddToP1);

      void Ressample(cBasicGeomCap3D * aG1,EpipolaireCoordinate & E1,double aStep);
};

Pt2dr  cApply_CreateEpip_main::DirEpipIm2(cBasicGeomCap3D * aG1,cBasicGeomCap3D * aG2,int aNbGlob,ElPackHomologue & aPack,bool AddToP1)
{
    Pt2dr aSz =  Pt2dr(aG1->SzBasicCapt3D());
 
    int aNbZ = 2;

    Pt2dr aSomTens2(0,0);

    double aIProf = aG1->GetVeryRoughInterProf();

    double aEps = 5e-4;

    double aLenghtSquare = ElMin(mLengthMax,sqrt((aSz.x*aSz.y) / (aNbGlob*aNbGlob)));


    int aNbX = ElMax(1+2*mDegre,round_up(aSz.x /aLenghtSquare));
    int aNbY = ElMax(1+2*mDegre,round_up(aSz.y /aLenghtSquare));


    std::cout << "NBBBB " << aNbX << " " << aNbY << "\n";
     

    for (int aKX=0 ; aKX<= aNbX ; aKX++)
    {
        double aPdsX = ElMax(aEps,ElMin(1-aEps,aKX /double(aNbX)));
        for (int aKY=0 ; aKY<= aNbY ; aKY++)
        {
            double aPdsY = ElMax(aEps,ElMin(1-aEps,aKY/double(aNbY)));
            Pt2dr aPIm1 = aSz.mcbyc(Pt2dr(aPdsX,aPdsY));
            if (aG1->CaptHasData(aPIm1))
            {
                Pt3dr aPT1;
                Pt3dr aC1;
                aG1->GetCenterAndPTerOnBundle(aC1,aPT1,aPIm1);

                std::vector<Pt2dr> aVPIm2;
                for (int aKZ = -aNbZ ; aKZ <= aNbZ ; aKZ++)
                {
                     Pt3dr aPT2 = aC1 + (aPT1-aC1) * (1+(aIProf*aKZ) / aNbZ);
                     if (aG1->PIsVisibleInImage(aPT2) && aG2->PIsVisibleInImage(aPT2))
                     {
                        aVPIm2.push_back(aG2->Ter2Capteur(aPT2));
                        ElCplePtsHomologues aCple(aPIm1,aVPIm2.back(),1.0);
                        if (! AddToP1) 
                           aCple.SelfSwap();
                        aPack.Cple_Add(aCple);
                     }
                }
                if (aVPIm2.size() >=2)
                {
                    Pt2dr aDir2 = vunit(aVPIm2.back()-aVPIm2[0]);
                    aSomTens2 = aSomTens2 + aDir2 * aDir2; // On double l'angle pour en faire un tenseur
                }
            }
        }
    }
    Pt2dr aRT = Pt2dr::polar(aSomTens2,0.0);
    return Pt2dr::FromPolar(1.0,aRT.y/2.0);
}


/*
void cApply_CreateEpip_main::Ressample(cBasicGeomCap3D * aG1,EpipolaireCoordinate & anE,double aStep)
{
}
*/




void cApply_CreateEpip_main::DoEpipGen()
{
      mDegre = 5;
      mLengthMax = 500.0;
      ElPackHomologue aPack;
      Pt2dr aDir2 =  DirEpipIm2(mGenI1,mGenI2,50,aPack,true);
      Pt2dr aDir1 =  DirEpipIm2(mGenI2,mGenI1,50,aPack,false);

      std::cout << "Compute Epip\n";
      CpleEpipolaireCoord * aCple = CpleEpipolaireCoord::PolynomialFromHomologue(aPack,mDegre,aDir1,aDir2);

      EpipolaireCoordinate & e1 = aCple->EPI1();
      EpipolaireCoordinate & e2 = aCple->EPI2();

      Pt2dr aInf1(1e20,1e20),aSup1(-1e20,-1e20);
      Pt2dr aInf2(1e20,1e20),aSup2(-1e20,-1e20);

      double aErrMax = 0.0;
      double aErrMoy = 0.0;
      int    mNbP = 0;

      for (ElPackHomologue::const_iterator itC=aPack.begin() ; itC!= aPack.end() ; itC++)
      {
           Pt2dr aP1 = e1.Direct(itC->P1());
           Pt2dr aP2 = e2.Direct(itC->P2());
           aInf1 = Inf(aInf1,aP1);
           aSup1 = Sup(aSup1,aP1);
           aInf2 = Inf(aInf2,aP2);
           aSup2 = Sup(aSup2,aP2);

           double anErr = ElAbs(aP1.y-aP2.y);
           mNbP++;
           aErrMax = ElMax(anErr,aErrMax);
           aErrMoy += anErr;
      }


      std::cout << aInf1 << " " <<  aSup1 << "\n";
      std::cout << aInf2 << " " <<  aSup2 << "\n";
      std::cout << "DIR " << aDir1 << " " << aDir2 << "\n";
      std::cout << "Epip Rect Accuracy, Moy " << aErrMoy/mNbP << " Max " << aErrMax << "\n";
}





cApply_CreateEpip_main::cApply_CreateEpip_main(int argc,char ** argv)
{
    Tiff_Im::SetDefTileFile(50000);
    std::string aDir= ELISE_Current_DIR;
    std::string aName1;
    std::string aName2;
    std::string anOri;
    double  aScale=1.0;

    bool Gray = true;
    bool Cons16B = true;
    bool InParal = true;
    bool DoIm = true;
    std::string aNameHom;
    int  aDegre = -1;


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aName1,"Name first image", eSAM_IsExistFile)
                << EAMC(aName2,"Name second image", eSAM_IsExistFile)
                << EAMC(anOri,"Name orientation", eSAM_IsExistDirOri),
    LArgMain()  << EAM(aScale,"Scale",true)
                << EAM(aDir,"Dir",true,"directory (Def=current)", eSAM_IsDir)
                    << EAM(Gray,"Gray",true,"One channel gray level image (Def=true)", eSAM_IsBool)
                    << EAM(Cons16B,"16B",true,"Maintain 16 Bits images if avalaible (Def=true)", eSAM_IsBool)
                    << EAM(InParal,"InParal",true,"Compute in parallel (Def=true)", eSAM_IsBool)
                    << EAM(DoIm,"DoIm",true,"Compute image (def=true !!)", eSAM_IsBool)
                    << EAM(aNameHom,"NameH",true,"Extension to compute Hom point in epi coord (def=none)", eSAM_NoInit)
                    << EAM(aDegre,"Degre",true,"Degre of polynom to correct epi (def=1-, ,2,3)")
    );

    if (!MMVisualMode)
    {
    if (aName1 > aName2) ElSwap(aName1,aName2);



    int aNbChan = Gray ? 1 : - 1;

    cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc
                                               (
                                                   argc,argv,
                                                   aDir,
                                                   aTplFCND
                                               );
     mICNM->CorrecNameOrient(anOri);


     mGenI1 = mICNM->StdCamGenOfNames(anOri,aName1);
     mGenI2 = mICNM->StdCamGenOfNames(anOri,aName2);

     if ((mGenI1->DownCastCS()==0) || (mGenI2->DownCastCS()==0))
     {
         DoEpipGen();
         return;
     }

     std::string   aKey =  + "NKS-Assoc-Im2Orient@-" + anOri;

     std::string aNameOr1 = mICNM->Assoc1To1(aKey,aName1,true);
     std::string aNameOr2 = mICNM->Assoc1To1(aKey,aName2,true);

     // std::cout << "RREEEEEEEEEEEEEEead cam \n";
     CamStenope * aCam1 = CamStenope::StdCamFromFile(true,aNameOr1,mICNM);
     // std::cout << "EPISZPPPpp " << aCam1->SzPixel() << "\n";

     CamStenope * aCam2 = CamStenope::StdCamFromFile(true,aNameOr2,mICNM);

     Tiff_Im aTif1 = Tiff_Im::StdConvGen(aDir+aName1,aNbChan,Cons16B);
     Tiff_Im aTif2 = Tiff_Im::StdConvGen(aDir+aName2,aNbChan,Cons16B);



      // aCam1->SetSz(aTif1.sz(),true);
      // aCam2->SetSz(aTif2.sz(),true);

// for (int aK=0; aK<13 ; aK++) std::cout << "SSSssssssssssssssssssiize !!!!\n"; getchar();

  //  Test commit


     cCpleEpip aCplE
               (
                    aDir,
                    aScale,
                    *aCam1,aName1,
                    *aCam2,aName2
               );

     const char * aCarHom = 0;
     if (EAMIsInit(&aNameHom))
        aCarHom = aNameHom.c_str();

     std::cout << "TimeEpi-0 \n";
     ElTimer aChrono;
     aCplE.ImEpip(aTif1,aNameOr1,true,InParal,DoIm,aCarHom,aDegre);
     std::cout << "TimeEpi-1 " << aChrono.uval() << "\n";
     aCplE.ImEpip(aTif2,aNameOr2,false,InParal,DoIm,aCarHom,aDegre);
     std::cout << "TimeEpi-2 " << aChrono.uval() << "\n";

     aCplE.SetNameLock("End");
     aCplE.LockMess("End cCpleEpip::ImEpip");


     return ;

    }
    else return ;
}


int CreateEpip_main(int argc,char ** argv)
{
     cApply_CreateEpip_main(argc,argv);

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
