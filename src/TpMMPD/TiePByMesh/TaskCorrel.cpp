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
        int SzPtCorr = 1;double corl_seuil_glob = 0.8;
        int SzAreaCorr = 5; double corl_seuil_pt = 0.9;
        double PasCorr=0.5;
        vector<string> dParam; dParam.push_back("NO");
        bool useExistHomoStruct = false;
        double aAngleF = 90;
        vector<double> aParamD;
        string aDirXML = "XML_TiepTri";
        bool Test=false;
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
                    << EAM(aDirXML, "OutXML", true, "Output directory for XML File. Default = XML_TiepTri")
                    << EAM(Test, "Test", true, "Test stretching")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        if (!Test)
        {
            InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                              SzPtCorr, SzAreaCorr,
                                              corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
            aChain->initAll(pathPlyFileS);
            vector<pic*> PtrPic = aChain->getmPtrListPic();
            vector<triangle*> PtrTri = aChain->getmPtrListTri();
            cout<<"Process has " <<PtrPic.size()<<" PIC && "<<PtrTri.size()<<" TRI "<<endl;
            CorrelMesh aCorrel(aChain);

            vector<cXml_TriAngulationImMaster> lstJobTriAngulationImMaster;
            aChain->creatJobCorrel(aAngleF , lstJobTriAngulationImMaster);

            cout<<"There is "<<lstJobTriAngulationImMaster.size()<<" jobs xml"<<endl;
            for (uint i=0; i<lstJobTriAngulationImMaster.size(); i++)
            {
                cXml_TriAngulationImMaster aTriAngulationImMaster =  lstJobTriAngulationImMaster[i];
                for (uint ii=0; ii<aTriAngulationImMaster.Tri().size(); ii++)
                {
                    cXml_Triangle3DForTieP aTriangle3DForTieP = aTriAngulationImMaster.Tri()[ii];
                }
            }

            ELISE_fp::MkDirSvp(aDirXML);
            for (uint aK=0; aK<lstJobTriAngulationImMaster.size(); aK++)
            {
                string fileXML =  aChain->getPrivmICNM()->Dir() + aDirXML + "/" + lstJobTriAngulationImMaster[aK].NameMaster() + ".xml";
                MakeFileXML(lstJobTriAngulationImMaster[aK], fileXML);
            }
            cout<<endl;
        }
        else
        {
            InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                              SzPtCorr, SzAreaCorr,
                                              corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
            aChain->initAll(pathPlyFileS);
            vector<pic*> PtrPic = aChain->getmPtrListPic();
            vector<triangle*> PtrTri = aChain->getmPtrListTri();
            cout<<"Process has " <<PtrPic.size()<<" PIC && "<<PtrTri.size()<<" TRI "<<endl;


            int aInd =0;
            int aIndp=0;
            triangle * atri = PtrTri[aInd];
            pic * apic = PtrPic[aIndp];
            cElPlan3D * aPlan = new cElPlan3D(atri->getSommet(0) , atri->getSommet(1), atri->getSommet(2));
            ElRotation3D    aRot_PE = aPlan->CoordPlan2Euclid();
            ElMatrix<REAL>  aRot_EP = aRot_PE.inv().Mat();
            ElMatrix<REAL>  aPt(3,3);
            for (uint akPt=0; akPt<3; akPt++)
            {
                aPt(akPt,0) = atri->getSommet(akPt).x;
                aPt(akPt,1) = atri->getSommet(akPt).y;
                aPt(akPt,2) = atri->getSommet(akPt).z;
            }
            ElMatrix<REAL> aPtP(3,3);
            aPtP = aRot_EP*aPt;
            cout<<"Euclid: "<<atri->getSommet(0)<<atri->getSommet(1)<<atri->getSommet(2)<<endl;
            Pt3dr aPt0P;Pt3dr aPt1P;Pt3dr aPt2P;
            aPtP.GetCol(0, aPt0P);
            aPtP.GetCol(1, aPt1P);
            aPtP.GetCol(2, aPt2P);
            cout<<"Plan: "<<aPt0P<<aPt1P<<aPt2P<<endl;
            ElAffin2D aAffPlan2Img (ElAffin2D::Id());
            Tri2d aPtT = *atri->getReprSurImg()[apic->mIndex];
            aAffPlan2Img = ElAffin2D::FromTri2Tri
                             (
                                  Pt2dr(aPt0P.x, aPt0P.y),
                                  Pt2dr(aPt1P.x, aPt1P.y),
                                  Pt2dr(aPt2P.x, aPt2P.y),
                                  aPtT.sommet1[0],
                                  aPtT.sommet1[1],
                                  aPtT.sommet1[2]
                             );

             cout<<"Aff :"<<aAffPlan2Img.I01()<<aAffPlan2Img.I10()<<aAffPlan2Img.I00()<<endl;

             ElMatrix<REAL> affineVerif(2,2);
             ElMatrix<REAL> ptVerif (1,2);

             affineVerif(0,0) = aAffPlan2Img.I01().x;
             affineVerif(0,1) = aAffPlan2Img.I01().y;
             affineVerif(1,0) = aAffPlan2Img.I10().x;
             affineVerif(1,1) = aAffPlan2Img.I10().y;

             ptVerif(0,0) = aPt0P.x;
             ptVerif(0,1) = aPt0P.y;

             ElMatrix<REAL> ptVerifAff(1,1);
             ptVerifAff = affineVerif*ptVerif;


             cout<<"Verif : ptPlan :" <<aPt0P<<" - PtIm :"<<aAffPlan2Img(Pt2dr(aPt0P.x, aPt0P.y))<<" -PtImOrg :"<<aPtT.sommet1[0]<<endl;
             cout<<"Verif : ptPlan :" <<aPt0P<<" - PtIm :"<<Pt2dr(ptVerifAff(0,0) , ptVerifAff(0,1))<<" -PtImOrg :"<<aPtT.sommet1[0]<<endl;



        }




        return EXIT_SUCCESS;
    }
