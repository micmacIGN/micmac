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
#include "../include/im_tpl/image.h"

/*
*/

bool IsGoodVal(const double & aV)
{
	return (!std_isnan(aV)) && (!std_isinf(aV));
}
bool IsGoodVal(const Pt2dr & aP)
{
    return IsGoodVal(aP.x) && IsGoodVal(aP.y);
}

bool IsGoodVal(const Pt3dr & aP)
{
    return IsGoodVal(aP.x) && IsGoodVal(aP.y) && IsGoodVal(aP.z) ;
}


/********************************************************/
/*                                                      */
/*                    GESTION DES ERREURS               */
/*                                                      */
/********************************************************/

void MakeFileJunk(const std::string & aName)
{
     std::cout << "ERREUR SUR " << aName << "\n";
     std::string JUNK_PREF = "MMJunk";
     std::string aDest = DirOfFile(aName) + JUNK_PREF + "-" + NameWithoutDir(aName) + "." + JUNK_PREF;
     ELISE_fp::MvFile ( aName, aDest);
}

class cTetstFileErrorHandler : public cElErrorHandlor
{
     public :
         cTetstFileErrorHandler(const std::string & aName) :
             mName (aName)
         {
             if (! ELISE_fp::exist_file(aName))
             {
                  std::cout << "Warn  " << aName << " does not exist \n";
                  exit(EXIT_SUCCESS);
             }
         }

         void OnError()
         {
              MakeFileJunk(mName);
              exit(EXIT_SUCCESS);
         }

         std::string mName;
};

void InitJunkErrorHandler(const std::string & aName)
{
    TheCurElErrorHandlor = new cTetstFileErrorHandler(aName);
}

std::string InitJunkErrorHandler(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);
    if (argc <  2)
    {
        std::cout << "Warn not enough arg \n";
        exit(EXIT_SUCCESS);
    }
    std::string aName = argv[1];
    TheCurElErrorHandlor = new cTetstFileErrorHandler(aName);

    return aName;
}

void CheckSetFile(const std::string & aDir,const std::string & aKey,const std::string & aKeyCom)
{
    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const std::vector<std::string> * aVName = aICNM->Get(aKey);

    std::list<std::string> aLCom;
    for (int aK=0; aK< int (aVName->size()) ; aK++)
    {
        std::string aCom =  MM3dBinFile(" TestLib " + aKeyCom ) + aDir+(*aVName)[aK];
        aLCom.push_back(aCom);
    }

     cEl_GPAO::DoComInParal(aLCom);
}


/********************************************************/
/*                                                      */
/*     Check Tiff                                       */
/*                                                      */
/********************************************************/

int CheckOneTiff_main(int argc,char ** argv)
{
   // Comme c'etait comme cela, j'y touche pas, mais rajoute quand meme qqchose
   std::string  aName = InitJunkErrorHandler(argc,argv);

   int  CorrecNan = 0;
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  << EAMC(aName,"Directory "),
        LArgMain()  << EAM(CorrecNan,"WCNan",true,"With check/correction of nan (1=check,2=correc)")
   );

   Tiff_Im aTF(aName.c_str());
   ELISE_COPY(aTF.all_pts(),aTF.in(),Output::onul());
   if (CorrecNan)
   {
      Im2D<REAL8,REAL8>  aIm =  Im2D<REAL8,REAL8>::FromFileStd(aName);
      TIm2D<REAL8,REAL8>  aTIm(aIm);
      
      Pt2di aSz = aIm.sz();
      Pt2di aP;
      Im2D_Bits<1>       aMasqOk(aSz.x,aSz.y,1);
      TIm2DBits<1>      aTMasqOk(aMasqOk);
      Im2D_Bits<1>       aMasqRes(aSz.x,aSz.y,1);
      int aCptNan=0;
      for (aP.x=0 ; aP.x<aSz.x ;  aP.x++)
      {
          for (aP.y=0 ; aP.y<aSz.y ;  aP.y++)
          {
               REAL8  aV = aTIm.get(aP);
               if (std_isnan(aV))
               {
                  aCptNan++;
                  aTMasqOk.oset(aP,0);
                  aTIm.oset(aP,0.0);
               }
          }
      }
      std::cout << "Nb Nan = " << aCptNan << "\n";
      if (CorrecNan>=2)
      {
          Im2D<REAL8,REAL8> aImCorrec = ImpaintL2(aMasqOk,aMasqRes,aIm,4);

          std::string aNameRes = StdPrefix(aName)+"_Correc.tif";

          Tiff_Im aTifOut(aNameRes.c_str(),aSz,aTF.type_el(),aTF.mode_compr(),aTF.phot_interp());
          ELISE_COPY(aImCorrec.all_pts(),aImCorrec.in(),aTifOut.out());
      }
   }
   return EXIT_SUCCESS;
}

int CheckAllTiff_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string aDir;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory "),
        LArgMain()
    );

    CheckSetFile(aDir,"NKS-Set-TmpTifFile","Check1Tiff");

    return EXIT_SUCCESS;
}

/********************************************************/
/*                                                      */
/*     Check Hom                                        */
/*                                                      */
/********************************************************/


int CheckOneHom_main(int argc,char ** argv)
{
   std::string  aName = InitJunkErrorHandler(argc,argv);
   ElPackHomologue aPack= ElPackHomologue::FromFile(aName);

   for (ElPackHomologue::tCstIter itP=aPack.begin() ; itP!=aPack.end() ; itP++)
   {
       for (int aK=0 ; aK<itP->NbPts() ; aK++)
       {
           if (! IsGoodVal(itP->PK(aK)))
           {
               BasicErrorHandler();
           }
       }
   }
   return EXIT_SUCCESS;
}

int CheckAllHom_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string aDir;
    std::string aExt ="";
    std::string aPost = "dat";

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory "),
        LArgMain()  << EAM(aExt,"Extension ",true,"Like _SRes, Def=\"\"")
                    << EAM(aPost,"Extension ",true,"Post , Def = dat")
    );

    CheckSetFile(aDir,"NKS-Set-Homol@"+aExt+ "@"+aPost,"Check1Hom");

    return EXIT_SUCCESS;
}



/********************************************************/
/*                                                      */
/*     Check Orient                                     */
/*                                                      */
/********************************************************/

int CheckOneOrient_main(int argc,char ** argv)
{
   std::string  aName = InitJunkErrorHandler(argc,argv);

   CamStenope * aCam = BasicCamOrientGenFromFile(aName);

   if ((!IsGoodVal(aCam->Focale())) || (!IsGoodVal(aCam->PP())))
   {
      BasicErrorHandler();
   }
   return EXIT_SUCCESS;
}

int CheckAllOrient_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string aDir;
    std::string anOri ;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aDir,"Directory ")
                    << EAMC(anOri,"Orientation "),
        LArgMain()
    );

    StdCorrecNameOrient(anOri,aDir);

    CheckSetFile(aDir,"NKS-Set-Orient@-"+anOri,"Check1Ori");

    return EXIT_SUCCESS;
}


void ChekOneBigTiff(const std::string & aName,int aSz)
{
    Tiff_Im::SetDefTileFile(1<<30);
    // Pt2di aSz(aSz,Sz);
    Tiff_Im aTF(
            aName.c_str(),
            Pt2di(aSz,aSz),
            GenIm::u_int1,
            Tiff_Im::No_Compr,
            Tiff_Im::BlackIsZero
    );
    ELISE_COPY(aTF.all_pts(),1,aTF.out());

    double aSom;
    ELISE_COPY(aTF.all_pts(),Rconv(aTF.in()),sigma(aSom));
    double aDif = aSom- ElSquare(double(aSz));
    std::cout << " Som " << aSom << ";  Dif " << aDif  << "\n";
    ELISE_ASSERT(ElAbs(aDif)<1e-3,"ChekOneBigTiff");
}

int ChekBigTiff_main(int,char**)
{
    ChekOneBigTiff("ChekOneBigTiff_60000.tif",60000);
    ChekOneBigTiff("ChekOneBigTiff_65000.tif",65000);
    return EXIT_SUCCESS;
}

/********************************************************/
/*                                                      */
/*                    GESTION DES ERREURS               */
/*                                                      */
/********************************************************/








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
