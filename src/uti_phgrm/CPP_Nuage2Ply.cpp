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

#define DEF_OFSET -12349876


int Ratio(double aV1,double aV2)
{

   double aRes = aV1 / aV2;
   int aIRes = round_ni(aRes);
   if (ElAbs(aRes-aIRes) > 1e-2)
      return -1;

   return aIRes;
}

int Nuage2Ply_main(int argc,char ** argv)
{
    std::string aNameNuage,aNameOut,anAttr1;
    std::vector<string> aVCom;
    int aBin  = 1;
    std::string aMask;
    double aSeuilMask=1;

    int DoPly = 1;
    int DoXYZ = 0;
    int DoNrm = 0;

    double aSc=1.0;
    double aDyn = 1.0;
    double aExagZ = 1.0;
    Pt2dr  aP0(0,0);
    Pt2dr  aSz(-1,-1);
    double aRatio = 1.0;
    bool aDoMesh = false;
    bool DoublePrec = false;
    Pt3dr anOffset(0,0,0);


    ElInitArgMain
    (
    argc,argv,
    LArgMain()  << EAMC(aNameNuage,"Name of XML file", eSAM_IsExistFile),
    LArgMain()  << EAM(aSz,"Sz",true,"Sz (to crop)")
                    << EAM(aP0,"P0",true,"Origin (to crop)")
                    << EAM(aNameOut,"Out",true,"Name of result (default toto.xml => toto.ply)")
                    << EAM(aSc,"Scale",true,"Do change the scale of result (def=1, 2 mean smaller)")
                    << EAM(anAttr1,"Attr",true,"Image to colour the point", eSAM_IsExistFile)
                    << EAM(aVCom,"Comments",true,"Commentary to add in the ply file (Def=None)", eSAM_NoInit )
                    << EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
                    << EAM(aMask,"Mask",true,"Supplementary mask image", eSAM_IsExistFile)
                    << EAM(aSeuilMask,"SeuilMask", true, "Theshold for supplementary mask")
                    << EAM(aDyn,"Dyn",true,"Dynamic of attribute")
                    << EAM(DoPly,"DoPly",true,"Do Ply, def = true")
                    << EAM(DoXYZ,"DoXYZ",true,"Do XYZ, export as RGB image where R=X,G=Y,B=Z")
                    << EAM(DoNrm,"Normale",true,"Add normale (Def=false, usable for Poisson)")
                    << EAM(aExagZ,"ExagZ",true,"To exagerate the depth, Def=1.0")
                    << EAM(aRatio,"RatioAttrCarte",true,"")
                    << EAM(aDoMesh,"Mesh",true, "Do mesh (Def=false)")
                    << EAM(DoublePrec,"64B",true,"To generate 64 Bits ply, Def=false, WARN = do not work properly with meshlab or cloud compare")
                    << EAM(anOffset,"Offs", true, "Offset in points to limit 32 Bits accuracy problem")
    );

    if (!MMVisualMode)
    {
    if (EAMIsInit(&aSz))
    {
         std::cout << "Waaaarnnn  :  meaning of parameter has changed\n";
         std::cout <<  " it used to be the corner (this was a bug)\n";
         std::cout <<  " now it is really the size\n";
    }

    if (aNameOut=="")
      aNameOut = StdPrefix(aNameNuage) + ".ply";

    cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromFileIm(aNameNuage,"XML_ParamNuage3DMaille","",aExagZ);
    if (aMask !="")
    {
         Im2D_Bits<1> aMaskN= aNuage->ImDef();
         Tiff_Im aMaskSup(aMask.c_str());
         ELISE_COPY(aMaskN.all_pts(),(aMaskSup.in(0) >= aSeuilMask) && aMaskN.in(), aMaskN.out());
    }

    if (aSz.x <0)
        aSz = Pt2dr(aNuage->SzUnique());

    if ( ( anAttr1.length()!=0 ) && ( !ELISE_fp::exist_file( anAttr1 ) ) )
    {
        cerr << "ERROR: colour image [" << anAttr1 << "] does not exist" << endl;
        return EXIT_FAILURE;
    }

    if (anAttr1!="")
    {
       anAttr1 = NameFileStd(anAttr1,3,false,true,true,true);
       // std::cout << "ATTR1 " << anAttr1 << "\n";

       if (! EAMIsInit(&aRatio))
       {
            Pt2dr aSzNuage = Pt2dr(aNuage->SzUnique());
            Pt2dr aSzImage = Pt2dr(Tiff_Im(anAttr1.c_str()).sz());

            int aRx = Ratio(aSzImage.x,aSzNuage.x);
            int aRy = Ratio(aSzImage.y,aSzNuage.y);
            if ((aRx==aRy) && (aRx>0))
            {
               aRatio = aRx;
            }
            else
            {
               aRatio = 1;
               std::cout << "WARnnnnnnnnnnnn\n";
               std::cout << "Cannot get def value of RatioAttrCarte, set it to 1\n";
            }
            //std::cout << "RR " << aRx <<  " " << aRy << " SZss " << aSzNuage << aSzImage << "\n"; getchar();
       }
       aNuage->Std_AddAttrFromFile(anAttr1,aDyn,aRatio);
    }

     cElNuage3DMaille * aRes = aNuage->ReScaleAndClip(Box2dr(aP0,aP0+aSz),aSc);
     //cElNuage3DMaille * aRes = aNuage;
    std::list<std::string > aLComment(aVCom.begin(), aVCom.end());

    if (DoPly)
    {

       if (aDoMesh)
       {
           aRes->AddExportMesh();
       }

        aRes->PlyPutFile( aNameOut, aLComment, (aBin!=0), DoNrm, DoublePrec, anOffset );
    }
    if (DoXYZ)
    {
        aRes->NuageXZGCOL(StdPrefix(aNameNuage),DoublePrec);
    }

    cElWarning::ShowWarns(DirOfFile(aNameNuage)  + "WarnNuage2Ply.txt");
    BanniereMM3D();

    return EXIT_SUCCESS;

    }
    else return EXIT_SUCCESS;
}


int PlySphere_main(int argc,char ** argv)
{
    Pt3dr aC;
    Pt3di aCoul(255,0,0);
    double aRay;
    int aNbPts=5;
    std::string Out="Sphere.ply";


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aC,"Center of sphere")
                    << EAMC(aRay,"Ray of sphere"),
        LArgMain()  << EAM(aNbPts,"NbPts",true,"Number of Pts / direc (Def=5, give 1000 points)")
    );

    std::vector<Pt3di> aVCol;
    std::vector<Pt3dr> aVpt;
    for (int anX=-aNbPts; anX<=aNbPts ; anX++)
    {
       for (int anY=-aNbPts; anY<=aNbPts ; anY++)
       {
          for (int aZ=-aNbPts; aZ<=aNbPts ; aZ++)
          {
               Pt3dr aP(anX,anY,aZ);
               aP = aP * (aRay/aNbPts);
               if (euclid(aP) <= aRay)
               {
                  aVpt.push_back(aC+aP);
                  aVCol.push_back(aCoul);
               }
          }
       }
    }
    std::list<std::string> aVCom;
    std::vector<const cElNuage3DMaille *> aVNuage;
    cElNuage3DMaille::PlyPutFile
    (
          Out,
          aVCom,
          aVNuage,
          &aVpt,
          &aVCol,
          true
    );

    return 1;
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
