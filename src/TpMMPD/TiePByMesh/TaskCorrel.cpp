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

#include "InitOutil.h"
#include "CorrelMesh.h"
#include "Pic.h"
#include "Triangle.h"
#include <stdio.h>

    /******************************************************************************
    The main function.
    ******************************************************************************/
int TaskCorrel_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    TaskCorrel - creat XML for TiepTri                *"<<endl;
    cout<<"********************************************************"<<endl;
        string pathPlyFileS ;
        string aTypeD="HOMOLINIT";
        string aFullPattern, aOriInput;
        string aHomolOut = "_Filtered";
        bool assum1er=false;
        int SzPtCorr = 1;int indTri=-1;double corl_seuil_glob = 0.8;bool Test=false;
        int SzAreaCorr = 5; double corl_seuil_pt = 0.9;
        double PasCorr=0.5;
        vector<string> dParam; dParam.push_back("NO");
        bool useExistHomoStruct = false;
        double aAngleF = 90;
        bool debugByClick = false;
        vector<double> aParamD;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                    << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                    << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                    //optional arguments
                    LArgMain()
                    << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                    << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                    << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                          SzPtCorr, SzAreaCorr,
                                          corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
        aChain->initAll(pathPlyFileS);
        vector<pic*> PtrPic = aChain->getmPtrListPic();
        vector<triangle*> PtrTri = aChain->getmPtrListTri();
        cout<<"Process has " <<PtrPic.size()<<" PIC && "<<PtrTri.size()<<" TRI "<<endl;
        CorrelMesh aCorrel(aChain);

        cout<<"********** Method TEST **********"<<endl;
        vector<cXml_TriAngulationImMaster> lstJobTriAngulationImMaster;
        aChain->creatJobCorrel(aAngleF , lstJobTriAngulationImMaster);



        cout<<"There is "<<lstJobTriAngulationImMaster.size()<<" jobs xml"<<endl;
        for (uint i=0; i<lstJobTriAngulationImMaster.size(); i++)
        {
            cXml_TriAngulationImMaster aTriAngulationImMaster =  lstJobTriAngulationImMaster[i];

            cout<<"Master = "<<aTriAngulationImMaster.NameMaster()<<endl;

            cout<<"Img2nd = "<<endl;
            for (uint j=0; j<aTriAngulationImMaster.NameSec().size(); j++)
                  cout<<"   ++ "<<aTriAngulationImMaster.NameSec()[j]<<endl;

            cout<<"There is "<<aTriAngulationImMaster.Tri().size()<<" tri get same master : "<<endl;
            for (uint ii=0; ii<aTriAngulationImMaster.Tri().size(); ii++)
            {
                cXml_Triangle3DForTieP aTriangle3DForTieP = aTriAngulationImMaster.Tri()[ii];
                for (uint kk=0; kk<aTriangle3DForTieP.NumImSec().size(); kk++)
                    cout<<aTriangle3DForTieP.NumImSec()[kk]<<" ";
            }
            cout<<endl;
        }
        MakeFileXML(lstJobTriAngulationImMaster[1],"TriTest.xml");

        cout<<endl;




        return EXIT_SUCCESS;
    }
