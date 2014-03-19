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

/*



Bascule ".*.jpg" RadialExtended B2   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=90
Bascule ".*.jpg" RadialExtended R2.xml   ImRep=00000002.jpg  P1Rep=[463,1406] P2Rep=[610,284] Teta=45


Bascule ".*.jpg" RadialExtended B2 MesureIm=OutAligned-Im.xml Teta=90
Tarama ".*.jpg" B2

Bascule ".*.jpg" RadialExtended R2.xml MesureIm=OutAligned.xml Teta=180
Tarama ".*.jpg" RadialExtended Repere=R2.xml

Bascule ".*.jpg" RadialExtended R3.xml MesureIm=OutAligned.xml Teta=180

*/

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
    bool ExpTxt=false;
    std::string PostPlan="_Masq";
    std::string KeyCalcMasq;
    std::string KeyEquivNoMasq;
    std::string MasqGlob;
    double  aResol=10;
    bool AcceptNoMask;
    std::string aPostIn= "";
    std::string aPostOut= "MasqFiltered";
    


    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)" ),
	LArgMain()  
                    << EAM(PostPlan,"PostPlan",true,"Post to plan, Def : toto ->toto_Masq.tif like with SaisieMasq")
                    << EAM(MasqGlob,"GlobalMasq",true,"Global Masq to add to all image")	
                    << EAM(KeyCalcMasq,"KeyCalculMasq",true,"For tuning masq per image")	
                    << EAM(KeyEquivNoMasq,"KeyEquivNoMasq",true,"When given if KENM(i1)==KENM(i2), don't masq")	
                    << EAM(aResol,"Resol",true,"Sub Resolution for masq storing, Def=10")
                    << EAM(AcceptNoMask,"ANM",true,"Accept no mask, def = true if MasqGlob and false else")
                    << EAM(ExpTxt,"ExpTxt",true,"Ascii fomat for in and out, def=false")
                    << EAM(aPostIn,"PostIn",true,"Post for Input dir Hom, Def=")
                    << EAM(aPostOut,"PostOut",true,"Post for Output dir Hom, Def=MasqFiltered")

    );
	
    #if (ELISE_windows)
		replace( aFullDir.begin(), aFullDir.end(), '\\', '/' );
     #endif
    SplitDirAndFile(aDir,aPat,aFullDir);
    if (EAMIsInit(&PostPlan))
    {
        CorrecNameMasq(aDir,aPat,PostPlan);
    }

    if (!EAMIsInit(&AcceptNoMask))
       AcceptNoMask = EAMIsInit(&MasqGlob);


    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);


    Im2D_Bits<1>  aImMasqGlob(1,1);
    if (EAMIsInit(&MasqGlob))
       aImMasqGlob = GetMasqSubResol(aDir+MasqGlob,aResol);


    const std::vector<std::string> *  aVN = anICNM->Get(aPat);
    std::vector<Im2D_Bits<1> >  aVMasq;

    for (int aKN = 0 ; aKN<int(aVN->size()) ; aKN++)
    {
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
    }

    std::string anExt = ExpTxt ? "txt" : "dat";


    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aPostIn)
                       +  std::string("@")
                       +  std::string(anExt);
    std::string aKHOut =   std::string("NKS-Assoc-CplIm2Hom@")
                        +  std::string(aPostOut)
                        +  std::string("@")
                       +  std::string(anExt);




    for (int aKN1 = 0 ; aKN1<int(aVN->size()) ; aKN1++)
    {
        for (int aKN2 = 0 ; aKN2<int(aVN->size()) ; aKN2++)
        {
             std::string aNameIm1 = (*aVN)[aKN1];
             std::string aNameIm2 = (*aVN)[aKN2];

             std::string aNameIn = aDir + anICNM->Assoc1To2(aKHIn,aNameIm1,aNameIm2,true);

             if (ELISE_fp::exist_file(aNameIn))
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
                  for (ElPackHomologue::const_iterator itP=aPackIn.begin(); itP!=aPackIn.end() ; itP++)
                  {
                      Pt2dr aP1 = itP->P1();
                      Pt2dr aP2 = itP->P2();
                      Pt2di aQ1 = round_ni(aP1/aResol);
                      Pt2di aQ2 = round_ni(aP2/aResol);

                      if ((aMasq1.get(aQ1,0) && aMasq2.get(aQ2,0)) || (! UseMasq))
                      {
                          ElCplePtsHomologues aCple(aP1,aP2);
                          aPackOut.Cple_Add(aCple);
                      }
                  }
                  std::string aNameOut = aDir + anICNM->Assoc1To2(aKHOut,aNameIm1,aNameIm2,true);
                  aPackOut.StdPutInFile(aNameOut);
                  std::cout << "IN " << aNameIn << " " << aNameOut  << " UseM " << UseMasq << "\n";
             }
        }
    }
    // std::vector<cImFMasq *> mVIm;
    

   
   return 0;
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
