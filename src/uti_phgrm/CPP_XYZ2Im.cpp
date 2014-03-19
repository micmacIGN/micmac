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

/*
*/





int TransfoCam_main(int argc,char ** argv,bool Ter2Im)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string  aFullNC,aFilePtsIn,aFilePtsOut;
    std::string XYZ = "X,Y,Z";
    std::string IJ = "I,J";

    ElInitArgMain
    (
           argc,argv,
           LArgMain()  <<  EAMC(aFullNC,"Nuage or Cam")
                      << EAMC(aFilePtsIn,"File In : " + (Ter2Im ? XYZ : IJ))
                      << EAMC(aFilePtsOut,"Out File : " + (Ter2Im ? IJ : XYZ)),
           LArgMain() 
    );

    std::string aDir,aNC;

    SplitDirAndFile(aDir,aNC,aFullNC);

    cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    cResulMSO aRMso =  anICNM->MakeStdOrient(aNC,false);


    cElNuage3DMaille *  aNuage = aRMso.Nuage();
    ElCamera         * aCam    = aRMso.Cam();
    if (! Ter2Im)
    {
        if (aNuage)
        {
            std::cout  << "For name " << aFullNC << "\n";
            ELISE_ASSERT(aNuage!=0,"Is not a MicMac Cloud -XML specif");
        }
    }

    ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);
    FILE *  aFOut = FopenNN(aFilePtsOut.c_str(),"w","XYZ2Im");

    char * aLine;

    while ((aLine = aFIn.std_fgets()))
    {
        if (Ter2Im)
        {
            Pt3dr aP;
            int aNb = sscanf(aLine,"%lf %lf %lf",&aP.x,&aP.y,&aP.z);
            ELISE_ASSERT(aNb==3,"Could not read 3 double values");
         
            Pt2dr aPIm;
            if (aNuage) aPIm = aNuage->Terrain2Index(aP);
            if (aCam)   aPIm = aCam->R3toF2(aP);
 
            fprintf(aFOut,"%lf %lf\n",aPIm.x,aPIm.y);
        }
        else
        {
            Pt2dr aPIm;
            int aNb = sscanf(aLine,"%lf %lf",&aPIm.x,&aPIm.y);
            ELISE_ASSERT(aNb==2,"Could not read 2 double values");

            if (aNuage->CaptHasData(aPIm))
            {
               Pt3dr aP  = aNuage->PreciseCapteur2Terrain(aPIm);
               fprintf(aFOut,"%lf %lf %f\n",aP.x,aP.y,aP.z);
            }
            else
            {
                std::cout << "Warn :: " << aPIm << " has no data in cloud\n";
            }
        }
     }


    aFIn.close();
    ElFclose(aFOut);

    return 0;
}


int XYZ2Im_main(int argc,char ** argv)
{
    return TransfoCam_main(argc,argv,true);
}

int Im2XYZ_main(int argc,char ** argv)
{
    return TransfoCam_main(argc,argv,false);
}
/*
int main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string aFilePtsIn,aFilePtsOut;
    bool Help;
    eTypeFichierApp aType;

    std::string aStrType = argv[1];
    StdReadEnum(Help,aType,argv[1],eNbTypeApp);

    std::string aStrChSys;


    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aStrType,"Type of file") 
                      << EAMC(aFilePtsIn,"App File") ,
           LArgMain() << EAM(aFilePtsOut,"Out",true,"Xml Out File")
                      << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
    );



    cChSysCo * aCSC = 0;
    if (aStrChSys!="")
       aCSC = cChSysCo::Alloc(aStrChSys);

    if (aFilePtsOut=="")
    {
        aFilePtsOut =StdPrefixGen(aFilePtsIn) + ".xml";
    }

    ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);


    cDicoAppuisFlottant  aDico;
    char * aLine;
    int aCpt=0;
    std::vector<Pt3dr> aVPts;
    std::vector<std::string> aVName;
    while ((aLine = aFIn.std_fgets()))
    {
         if (aLine[0] != '#')
         {
            char aName[1000];
            char aTruc[1000];
            double anX,anY,aZ;

            int aNb=0;
            if (aType==eAppEgels)
            {
                aNb = sscanf(aLine,"%s %s %lf %lf %lf",aName,aTruc,&anX,&anY,&aZ);
                if (aNb!=5)
                {
                     std::cout <<  " At line " << aCpt << " of File "<< aFilePtsIn << "\n";
                     ELISE_ASSERT(false,"Could not read the 5 expected values");
                }
            }
            if (aType==eAppGeoCub)
            {
                aNb = sscanf(aLine,"%s %lf %lf %lf",aName,&anX,&anY,&aZ);
                if (aNb!=4)
                {
                     std::cout <<  " At line " << aCpt << " of File "<< aFilePtsIn << "\n";
                     ELISE_ASSERT(false,"Could not read the 4 expected values");
                }
            }
            Pt3dr aP(anX,anY,aZ);
            aVPts.push_back(aP);
            aVName.push_back(aName);
        }
        aCpt ++;
     }


    if (aCSC!=0)
    {
        aVPts = aCSC->Src2Cibl(aVPts);
    }


    for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
    {
        cOneAppuisDAF aOAD;
        aOAD.Pt() = aVPts[aKP];
        aOAD.NamePt() = aVName[aKP];
        aOAD.Incertitude() = Pt3dr(1,1,1);

        aDico.OneAppuisDAF().push_back(aOAD);
    }

    aFIn.close();
    MakeFileXML(aDico,aFilePtsOut);
}

*/





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
