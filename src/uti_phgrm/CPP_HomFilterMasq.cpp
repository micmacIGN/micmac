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
#include <algorithm>


#define DEF_OFSET -12349876

struct cImFMasq
{

};


Im2D_Bits<1>  GetMasqSubResol(const std::string & aName,double aResol)
{
    Tiff_Im aTF = Tiff_Im::BasicConvStd(aName);

     Pt2di aSzF = aTF.sz();
     Pt2di aSzR = round_ni(Pt2dr(aSzF)/aResol);

     Im2D_Bits<1> aRes(aSzR.x,aSzR.y);


     Fonc_Num aFonc = aTF.in_bool_proj();
     aFonc = StdFoncChScale(aFonc,Pt2dr(0,0),Pt2dr(aResol,aResol));
     aFonc = aFonc > 0.75;

     ELISE_COPY (aRes.all_pts(), aFonc,aRes.out());




     return aRes;
}



int HomFilterMasq_main(int argc,char ** argv)
{
    // MemoArg(argc,argv);
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;
    bool ExpTxtIn=false;
    bool ExpTxtOut=false;
    std::string PostPlan="_Masq";
    std::string KeyCalcMasq;
    std::string KeyEquivNoMasq;
    std::string MasqGlob;
    double  aResol=10;
    bool AcceptNoMask;
    std::string aPostIn= "";
    std::string aPostOut= "MasqFiltered";
    std::string aOriMasq3D,aNameMasq3D;
    cMasqBin3D * aMasq3D = 0;
    double  aDistId=-1;
    double  aDistHom=-1;
    bool    DoSym  = false;
    bool    DoCalNb = false;

    Pt2dr  aSelecTer;

    std::vector<double> aVParam3DRel;


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile),
        LArgMain()
                    << EAM(PostPlan,"PostPlan",true,"Post to plan, Def : toto ->toto_Masq.tif like with SaisieMasq")
                    << EAM(MasqGlob,"GlobalMasq",true,"Global Masq to add to all image")
                    << EAM(KeyCalcMasq,"KeyCalculMasq",true,"For tuning masq per image")
                    << EAM(KeyEquivNoMasq,"KeyEquivNoMasq",true,"When given if KENM(i1)==KENM(i2), don't masq")
                    << EAM(aResol,"Resol",true,"Sub Resolution for masq storing, Def=10")
                    << EAM(AcceptNoMask,"ANM",true,"Accept no mask, def = true if MasqGlob and false else")
                    << EAM(ExpTxtIn,"ExpTxt",true,"Ascii format for in and out, def=false")
                    << EAM(ExpTxtOut,"ExpTxtOut",true,"Ascii format for out when != in , def=ExpTxt")
                    << EAM(aPostIn,"PostIn",true,"Post for Input dir Hom, Def=")
                    << EAM(aPostOut,"PostOut",true,"Post for Output dir Hom, Def=MasqFiltered")
                    << EAM(aOriMasq3D,"OriMasq3D",true,"Orientation for Masq 3D")
                    << EAM(aNameMasq3D,"Masq3D",true,"File of Masq3D, Def=AperiCloud_${OriMasq3D}.ply")
                    << EAM(aVParam3DRel,"Param3DRel",true,"Relative 3D param [DZMin,DZMax,DistMax]")
                    << EAM(aSelecTer,"SelecTer",true,"[Per,Prop] Period of tiling on ground selection, Prop=proporion of selected")
                    << EAM(aDistId,"DistId",true,"Supress pair such that d(P1,P2) < DistId, def unused")
                    << EAM(aDistHom,"DistH",true,"Distance of reprojection for filtering homologous point")
                    << EAM(DoSym,"Symetrise",true,"Symetrise Files when dont exist")
                    << EAM(DoCalNb,"Nb",true,"Calculate number of homologous points")
    );
    bool aHasOri3D =  EAMIsInit(&aOriMasq3D);
    bool HasTerSelec = EAMIsInit(&aSelecTer);
    bool HasRel3DFilter = EAMIsInit(&aVParam3DRel);
    double aDZMin=0,aDZMax=0,aDistMax=0;
    if (HasRel3DFilter)
    {
       ELISE_ASSERT(aVParam3DRel.size()==3,"aVParam3DRel bad size");
       aDZMin = aVParam3DRel.at(0);
       aDZMax = aVParam3DRel.at(1);
       aDistMax = aVParam3DRel.at(2);
    }


    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
     #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    if (EAMIsInit(&PostPlan))
    {
        CorrecNameMasq(aDir,aPat,PostPlan);
    }


    if (!EAMIsInit(&AcceptNoMask))
       AcceptNoMask = EAMIsInit(&MasqGlob) || aHasOri3D || EAMIsInit(&aDistId) || DoSym;

    if (DoCalNb)
        AcceptNoMask = true;


    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

    std::string aKeyOri;
    if (aHasOri3D)
    {
        anICNM->CorrecNameOrient(aOriMasq3D);
        if (! EAMIsInit(&aNameMasq3D))
        {
              aNameMasq3D = aDir + "AperiCloud_" + aOriMasq3D + "_polyg3d.xml";
        }
        if (ELISE_fp::exist_file(aDir+aNameMasq3D))
        {
            aMasq3D = cMasqBin3D::FromSaisieMasq3d(aDir+aNameMasq3D);
        }
        else
        {
            ELISE_ASSERT(EAMIsInit(&aSelecTer) || (aDistHom>=0) || HasRel3DFilter ,"Unused OriMasq3D");
        }
        aKeyOri = "NKS-Assoc-Im2Orient@" + aOriMasq3D;
    }

    Im2D_Bits<1>  aImMasqGlob(1,1);
    if (EAMIsInit(&MasqGlob))
       aImMasqGlob = GetMasqSubResol(aDir+MasqGlob,aResol);


    const std::vector<std::string> *  aVN = anICNM->Get(aPat);
    std::vector<Im2D_Bits<1> >  aVMasq;

    std::vector<cBasicGeomCap3D *> aVCam;


    double aResolMoy = 0;

    for (int aKN = 0 ; aKN<int(aVN->size()) ; aKN++)
    {
        std::cout << "Prep Reste " << aVN->size() - aKN << "\n";
        std::string aNameIm = (*aVN)[aKN];
        Tiff_Im aTF = Tiff_Im::StdConvGen(aNameIm,1,false);
        Pt2di aSzG = aTF.sz();
        Pt2di aSzR (round_ni(Pt2dr(aSzG)/aResol));
        Im2D_Bits<1> aImMasq(aSzR.x,aSzR.y,1);


        std::string aNameMasq = StdPrefix(aNameIm)+PostPlan + ".tif";
        if (EAMIsInit(&KeyCalcMasq))
        {
            aNameMasq = anICNM->Assoc1To1(KeyCalcMasq,aNameIm,true);
        }

        if (ELISE_fp::exist_file(aNameMasq))
        {
            Im2D_Bits<1> aImMasqLoc = GetMasqSubResol(aDir+aNameMasq,aResol);
            ELISE_COPY(aImMasq.all_pts(),aImMasq.in() && aImMasqLoc.in(0),aImMasq.out());
        }
        else
        {
             if (!AcceptNoMask)
             {
                 std::cout << "For Im " << aNameIm << " file " << aNameMasq << " does not exist\n";
                 ELISE_ASSERT(false,"Masq not found");
             }
        }

        if (EAMIsInit(&MasqGlob))
        {

            ELISE_COPY(aImMasq.all_pts(),aImMasq.in() && aImMasqGlob.in(0),aImMasq.out());
        }

        aVMasq.push_back(aImMasq);
        // Tiff_Im::CreateFromIm(aImMasq,"SousRes"+aNameMasq);
        if (aHasOri3D)
        {
            // aVCam.push_back(anICNM->StdCamGenerikOfNames(aNameIm,aOriMasq3D));
            aVCam.push_back(anICNM->StdCamGenerikOfNames(aOriMasq3D,aNameIm));
            aResolMoy += aVCam.back()->GlobResol();
        }
    }
    if (aHasOri3D)
       aResolMoy /= aVCam.size();

    std::string anExtIn = ExpTxtIn ? "txt" : "dat";
    if (!EAMIsInit(&ExpTxtOut))
      ExpTxtOut = ExpTxtIn;
    std::string anExtOut = ExpTxtOut ? "txt" : "dat";


    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aPostIn)
                       +  std::string("@")
                       +  std::string(anExtIn);
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aPostOut)
                        +  std::string("@")
                       +  std::string(anExtOut);


    double aPeriodTer=0,aSeuilDistTer=0;
    if (HasTerSelec)
    {
       aPeriodTer = aSelecTer.x * aResolMoy;
       aSeuilDistTer = aPeriodTer * sqrt(aSelecTer.y);
    }

    double aNbInTer=0;
    double aNbTestTer=0;


    int aNbHomol=0;
    for (int aKN1 = 0 ; aKN1<int(aVN->size()) ; aKN1++)
    {
        std::cout << "Filter Reste " << aVN->size() - aKN1 << "\n";
        for (int aKN2 = 0 ; aKN2<int(aVN->size()) ; aKN2++)
        {
             std::string aNameIm1 = (*aVN)[aKN1];
             std::string aNameIm2 = (*aVN)[aKN2];

/*
if(MPD_MM())
{
std::cout << aNameIm1  << " # " << aNameIm2 << "\n";
}
*/

             std::string aNameIn = aDir + anICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);
             bool ExistFileIn =  ELISE_fp::exist_file(aNameIn);
             std::string aNameInSym = aDir + anICNM->Assoc1To2(aKHIn,aNameIm2,aNameIm1,true);
             bool SymThisFile = DoSym && (!ELISE_fp::exist_file(aNameInSym));

             if (ExistFileIn)
             {
                  bool UseMasq = true;
                  if (EAMIsInit(&KeyEquivNoMasq))
                  {
                       UseMasq =  (anICNM->Assoc1To1(KeyEquivNoMasq,aNameIm1,true) != anICNM->Assoc1To1(KeyEquivNoMasq,aNameIm2,true) );
                  }


                  TIm2DBits<1>  aMasq1 ( aVMasq[aKN1]);
                  TIm2DBits<1>  aMasq2 ( aVMasq[aKN2]);

                  ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
                  ElPackHomologue aPackOut;
                  ElPackHomologue aPackSymOut;
                  for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
                  {
                      aNbHomol++;
                      if (!DoCalNb)
                      {
                           Pt2dr aP1 = itP->P1();
                           Pt2dr aP2 = itP->P2();
                           Pt2di aQ1 = round_ni(aP1/aResol);
                           Pt2di aQ2 = round_ni(aP2/aResol);
                       
                           bool Ok = ((aMasq1.get(aQ1,0) && aMasq2.get(aQ2,0)) || (! UseMasq));
                       
                           if (Ok &&  aHasOri3D)
                           {
                               //  Pt3dr  aPTer= aVCam[aKN1]->PseudoInter(aP1,*(aVCam[aKN2]),aP2);
                               ElSeg3D aSeg1 = aVCam[aKN1]->Capteur2RayTer(aP1);
                               ElSeg3D aSeg2 = aVCam[aKN2]->Capteur2RayTer(aP2);
                               Pt3dr  aPTer= aSeg1.PseudoInter(aSeg2);
                               if (aMasq3D && (! aMasq3D->IsInMasq(aPTer)))
                                  Ok = false;
                       
                               if (Ok && HasTerSelec)
                               {
                                   bool OkTer =  (mod_real(aPTer.x,aPeriodTer) < aSeuilDistTer) && (mod_real(aPTer.y,aPeriodTer) < aSeuilDistTer);
                                   Ok = OkTer;
                                   aNbTestTer ++;
                                   aNbInTer += OkTer;
                               }
                               if (Ok && HasRel3DFilter)
                               {
                                   CamStenope * aCam1 = aVCam[aKN1]->DownCastCS();
                                   CamStenope * aCam2 = aVCam[aKN2]->DownCastCS();
                                   Pt3dr aC1 = aCam1->PseudoOpticalCenter();
                                   Pt3dr aC2 = aCam2->PseudoOpticalCenter();

                                   // double aDZ1   =  aC1.z-aPTer.z; MODIF MPD/Yann => ca semble plus logique dans l'autre sens
								   double aDZ1   =  aPTer.z-aC1.z;
                                   double aDist1 =  euclid(aC1-aPTer);
                                   // double aDZ2   =  aC2.z-aPTer.z;
								   double aDZ2   =  aPTer.z-aC2.z;
                                   double aDist2 =  euclid(aC2-aPTer);
								   
                                   if (  
                                           ((aDZ1<aDZMin) &&  (aDZ2<aDZMin))
                                        || ((aDZ2>aDZMax) &&  (aDZ2>aDZMax))
                                        || ((aDist1>aDistMax) &&  (aDist2>aDistMax))
                                      )
                                   {
                                      Ok = false;
                                   }
                               }
                       
                               if (Ok && (aDistHom >0 ))
                               {
                                   Pt2dr aRP1 =  aVCam[aKN1]->Ter2Capteur(aPTer);
                                   Pt2dr aRP2 =  aVCam[aKN2]->Ter2Capteur(aPTer);
                                   double aD1 = euclid(aP1,aRP1);
                                   double aD2 = euclid(aP2,aRP2);
                                   if ((aD1+aD2) > aDistHom)
                                   {
                                       Ok = false;
                                       std::cout << "DIST " << aD1 << " " << aD2 << "\n";
                                   }
                               }
                           }  
                       
                           if (Ok && (aDistId>=0))
                           {
                              if (euclid(aP1,aP2)<aDistId)
                                 Ok = false;
                           }
                       
                           if (Ok)
                           {
                               ElCplePtsHomologues aCple(aP1,aP2,itP->Pds());
                               aPackOut.Cple_Add(aCple);
                               if (SymThisFile) 
                               {
                                   ElCplePtsHomologues aCpleSym(aP2,aP1);
                                   aPackSymOut.Cple_Add(aCpleSym);
                               }
                           }
                      }
                  }

                  if (!DoCalNb)
                  {
                       std::string aNameOut = aDir + anICNM->Assoc1To2(aKHOut,aNameIm1,aNameIm2,true);
                       aPackOut.StdPutInFile(aNameOut);
                       if (SymThisFile)
                       {
                           std::string aNameInSymOut = aDir + anICNM->Assoc1To2(aKHOut,aNameIm2,aNameIm1,true);
                           aPackSymOut.StdPutInFile(aNameInSymOut);
                       }
                       // std::cout << "IN " << aNameIn << " " << aNameOut  << " UseM " << UseMasq  << " Nb=" <<  aPackOut.size() << "\n";
                  }
             }
        }
    }
    // std::vector<cImFMasq *> mVIm;

    if (DoCalNb)
    {
        std::cout << "Nb homol: " << aNbHomol << "\n";
    }
    else if (HasTerSelec)
    {
        std::cout << "A Posteriori Prop=" << aNbInTer / aNbTestTer << "\n";
    }



   return EXIT_SUCCESS;
}

int HomFusionPDVUnik_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv);
    std::string  aDir,aPat,aFullDir;
    std::string aPostIn= "";
    std::string aPostOut= "MasqFusion";
    bool ExpTxt=false;

    std::string aDir2;
    std::vector<std::string > aDirN;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(aDir2,"Dir of external point", eSAM_IsPatFile),
        LArgMain()  
                    << EAM(aPostIn,"PostIn",true,"Post for Input dir Hom, Def=")
                    << EAM(aPostOut,"PostOut",true,"Post for Output dir Hom, Def=MasqFusion")
                    << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
                    << EAM(aDirN,"DirN",true,"Supplementary dirs 2 merge")
    );

    #if (ELISE_windows)
        replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
     #endif
    cElemAppliSetFile anEASF(aFullDir);

    cInterfChantierNameManipulateur * anICNM = anEASF.mICNM;


    const std::vector<std::string> *  aVN = anEASF.SetIm();


    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aPostIn)
                       +  std::string("@")
                       +  std::string(anExt);
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aPostOut)
                        +  std::string("@")
                       +  std::string(anExt);


    aDirN.push_back(aDir);
    aDirN.push_back(aDir2);

    for (int aKN1 = 0 ; aKN1<int(aVN->size()) ; aKN1++)
    {
        for (int aKN2 = 0 ; aKN2<int(aVN->size()) ; aKN2++)
        {
             std::string aNameIm1 = (*aVN)[aKN1];
             std::string aNameIm2 = (*aVN)[aKN2];
             std::string aNameLocIn = aDir + anICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);
             ElPackHomologue aPackOut;
             int aNbH=0;

             for (int aKP=0 ; aKP<int(aDirN.size()) ; aKP++)
             {
                 std::string aNameIn=  aDirN[aKP] + aNameLocIn;
                 if (ELISE_fp::exist_file(aNameIn))
                 {
                     ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
                     aNbH++;
                     for (ElPackHomologue::const_iterator itP=aPackIn.begin() ; itP!=aPackIn.end() ; itP++)
                     {
                         aPackOut.Cple_Add(itP->ToCple());
                     }
                 }
             }
             if (aPackOut.size() !=0)
             {
                std::string aNameOut = aDir + anICNM->Assoc1To2(aKHOut,aNameIm1,aNameIm2,true);
                aPackOut.StdPutInFile(aNameOut);
                if (0)
                   std::cout << "aNbH " << aNbH << "\n";
             }
        }
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
