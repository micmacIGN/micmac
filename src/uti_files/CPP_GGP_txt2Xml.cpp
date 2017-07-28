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
// #include "XML_GEN/all_tpl.h"

/*
*/




class  cReadAppui : public cReadObject
{
    public :
        cReadAppui(char aComCar,const std::string & aFormat) :
               cReadObject(aComCar,aFormat,"S"),
               mPt(-1,-1,-1),
               mInc3(-1,-1,-1),
               mInc (-1)
        {
              AddString("N",&mName,true);
              AddPt3dr("XYZ",&mPt,true);
              AddDouble("Ix",&mInc3.x,false);
              AddDouble("Iy",&mInc3.y,false);
              AddDouble("Iz",&mInc3.z,false);
              AddDouble("I",&mInc,false);
        }

        std::string mName;
        Pt3dr       mPt;
        Pt3dr       mInc3;
        double      mInc;
};




int GCP_Txt2Xml_main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,3);

    std::string aFilePtsIn,aFilePtsOut;
    bool Help;
    eTypeFichierApp aType=eAppEgels;

    std::string aStrType;
    if (argc >=2)
    {
        aStrType = argv[1];
        StdReadEnum(Help,aType,argv[1],eNbTypeApp,true);

    }

    std::string aStrChSys;
    double aMul = 1.0;
    bool   aMulIncAlso = true;
	Pt3dr  aOffs(0,0,0);

    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aStrType,"Format specification", eSAM_None, ListOfVal(eNbTypeApp))
                      << EAMC(aFilePtsIn,"GCP  File", eSAM_IsExistFile),
           LArgMain() << EAM(aFilePtsOut,"Out",true,"Xml Out File")
                      << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
                      << EAM(aMul,"MulCo",true,"Multiplier of result (for development and testing use)",eSAM_InternalUse)
                      << EAM(aMulIncAlso,"MulInc",true,"Multiplier also incertitude ? (for development and testing use)",eSAM_InternalUse)
                      << EAM(aOffs,"Offs",true,"Offset to substruct to all coordinates ; Def=[0,0,0]")
    );

    // StdReadEnum(Help,aType,argv[1],eNbTypeApp,true);

    if (!MMVisualMode)
    {
        std::string aFormat;
        char        aCom;
        if (aType==eAppEgels)
        {
             aFormat = "N S X Y Z";
             aCom    = '#';
        }
        else if (aType==eAppGeoCub)
        {
             aFormat = "N X Y Z";
             aCom    = '%';
        }
        else if (aType==eAppInFile)
        {
           bool Ok = cReadObject::ReadFormat(aCom,aFormat,aFilePtsIn,true);
           ELISE_ASSERT(Ok,"File do not begin by format specification");
        }
        else if (aType==eAppXML)
        {
             aFormat = "00000";
             aCom    = '0';
           // bool Ok = cReadObject::ReadFormat(aCom,aFormat,aFilePtsIn,true);
           // ELISE_ASSERT(Ok,"File do not begin by format specification");
        }
        else
        {
            bool Ok = cReadObject::ReadFormat(aCom,aFormat,aStrType,false);
            ELISE_ASSERT(Ok,"Arg0 is not a valid format specif");
        }




        cTransfo3D * aCSC = 0;
        // cChSysCo * aCSC = 0;
        if (aStrChSys!="")
           aCSC = cTransfo3D::Alloc(aStrChSys,"");

        if (aFilePtsOut=="")
        {
            aFilePtsOut =StdPrefixGen(aFilePtsIn) + ".xml";
        }



        char * aLine;
        int aCpt=0;
        std::vector<Pt3dr> aVPts;
        std::vector<Pt3dr> aVInc;
        std::vector<std::string> aVName;


        if (aType==eAppXML)
        {
            if (aFilePtsOut==aFilePtsIn)
                aFilePtsOut = "GCPOut_"+aFilePtsIn;
            cDicoAppuisFlottant aD = StdGetObjFromFile<cDicoAppuisFlottant>
                                     (
                                          aFilePtsIn,
                                          StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                          "DicoAppuisFlottant",
                                          "DicoAppuisFlottant"
                                     );

              for
              (
                    std::list<cOneAppuisDAF>::iterator itA=aD.OneAppuisDAF().begin();
                    itA!=aD.OneAppuisDAF().end();
                    itA++
              )
              {
                  aVPts.push_back(itA->Pt());
                  aVInc.push_back(itA->Incertitude());
                  aVName.push_back(itA->NamePt());
              }
        }
        else
        {
            std::cout << "Comment=[" << aCom<<"]\n";
            std::cout << "Format=[" << aFormat<<"]\n";
            cReadAppui aReadApp(aCom,aFormat);
            ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);
            while ((aLine = aFIn.std_fgets()))
            {
                 if (aReadApp.Decode(aLine))
                 {
                    aVPts.push_back(aReadApp.mPt);
                    double  aInc = aReadApp.GetDef(aReadApp.mInc,1);
                    aVInc.push_back(aReadApp.GetDef(aReadApp.mInc3,aInc));
                    aVName.push_back(aReadApp.mName);
                }
                aCpt ++;
             }
            aFIn.close();
        }


        if (aCSC!=0)
        {
            aVPts = aCSC->Src2Cibl(aVPts);
        }

        if (aMul!=1.0)
        {
            for (int aK=0 ; aK<int(aVPts.size()) ; aK++)
            {
                 aVPts[aK] = aVPts[aK] * aMul;
                 if (aMulIncAlso)
                     aVInc[aK] = aVInc[aK] * aMul;
            }
        }


        cDicoAppuisFlottant  aDico;
        for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
        {
            cOneAppuisDAF aOAD;
            Pt3dr aPt;
            aPt.x = aVPts[aKP].x - aOffs.x;
            aPt.y = aVPts[aKP].y - aOffs.y;
            aPt.z = aVPts[aKP].z - aOffs.z;
            aOAD.Pt() = aPt;
            aOAD.NamePt() = aVName[aKP];
            aOAD.Incertitude() = aVInc[aKP];

            aDico.OneAppuisDAF().push_back(aOAD);
        }

        MakeFileXML(aDico,aFilePtsOut);

        return 0;
    }
    else
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
