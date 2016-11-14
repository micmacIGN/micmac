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

const double TT_SEUIL_SURF = 100;

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
            cout<<"Process has " <<PtrPic.size()<<" PIC && "<<PtrTri.size()<<" TRI "<<endl<<endl;

            double min_cur = DBL_MIN;
            pic * picMaster = NULL;
            //double cmptMas [int(PtrPic.size())];
            //Maybe this (compile error!!! at least on MS visual studio)
            vector<double> cmptMas(PtrPic.size());
            double cmptDel=0;
            for (uint acP = 0; acP<PtrPic.size(); acP++)
            {
                cmptMas[acP] = 0;
            }
            ElRotation3D    aRot_EP(ElRotation3D::Id);
            vector <ElAffin2D> VaAffLc2Im;
            for (uint aIT = 0; aIT<PtrTri.size(); aIT++)
            {
                //cout<<" + TRI : "<<aIT<<endl;
                min_cur = DBL_MIN;
                picMaster = NULL;
                for (uint aIP = 0; aIP<PtrPic.size(); aIP++)
                {
                    triangle * atri = PtrTri[aIT];
                    pic * apic = PtrPic[aIP];
                    Tri2d aTriIm = *atri->getReprSurImg()[apic->mIndex];
                    Pt2dr aPtI0 = aTriIm.sommet1[0];
                    Pt2dr aPtI1 = aTriIm.sommet1[1];
                    Pt2dr aPtI2 = aTriIm.sommet1[2];
                    double aSurf =  (aPtI0-aPtI1) ^ (aPtI0-aPtI2);
                    if (ElAbs(aSurf) > TT_SEUIL_SURF)
                    {
                        cElPlan3D * aPlanLoc = new cElPlan3D(atri->getSommet(0) , atri->getSommet(1), atri->getSommet(2));

                        ElRotation3D    aRot_PE = aPlanLoc->CoordPlan2Euclid();
                                        aRot_EP = aRot_PE.inv();

                        Pt3dr aPtP0 = aRot_EP.ImAff(atri->getSommet(0));
                        Pt3dr aPtP1 = aRot_EP.ImAff(atri->getSommet(1));
                        Pt3dr aPtP2 = aRot_EP.ImAff(atri->getSommet(2));

                        //cout<<"Euclid: "<<atri->getSommet(0)<<atri->getSommet(1)<<atri->getSommet(2)<<endl;
                        //cout<<"Image : "<<aPtI0<<aPtI1<<aPtI2<<endl;
                        //cout<<"PlanLoc:"<<aPtP0<<aPtP1<<aPtP2<<endl;



                        cElPlan3D * aPlanIm = new cElPlan3D (   Pt3dr(aPtI0.x, aPtI0.y, 0),
                                                                Pt3dr(aPtI1.x, aPtI1.y, 0),
                                                                Pt3dr(aPtI2.x, aPtI2.y, 0)   );
                        ElRotation3D aRot_EPIm = aPlanIm->CoordPlan2Euclid().inv();

                        Pt3dr aPtPIm0 = aRot_EPIm.ImAff(Pt3dr(aPtI0.x, aPtI0.y, 0));
                        Pt3dr aPtPIm1 = aRot_EPIm.ImAff(Pt3dr(aPtI1.x, aPtI1.y, 0));
                        Pt3dr aPtPIm2 = aRot_EPIm.ImAff(Pt3dr(aPtI2.x, aPtI2.y, 0) );



                        ElAffin2D aAffLc2Im;
                        aAffLc2Im = aAffLc2Im.FromTri2Tri(  Pt2dr(aPtP0.x, aPtP0.y),
                                                            Pt2dr(aPtP1.x, aPtP1.y),
                                                            Pt2dr(aPtP2.x, aPtP2.y),
                                                            Pt2dr(aPtPIm0.x, aPtPIm0.y),
                                                            Pt2dr(aPtPIm1.x, aPtPIm1.y),
                                                            Pt2dr(aPtPIm2.x, aPtPIm2.y)
                                                            );
                        VaAffLc2Im.push_back(aAffLc2Im);


                        //cout<<"PlanIm :"<<aPtPIm0<<aPtPIm1<<aPtPIm2<<endl;
                        //cout<<"Aff    :"<<aAffLc2Im.I10()<<aAffLc2Im.I01()<<aAffLc2Im.I00()<<endl;

                        double vecA_cr =  aAffLc2Im.I10().x*aAffLc2Im.I10().x + aAffLc2Im.I10().y*aAffLc2Im.I10().y;
                        double vecB_cr =  aAffLc2Im.I01().x*aAffLc2Im.I01().x + aAffLc2Im.I01().y*aAffLc2Im.I01().y;
                        double AB_cr   =  pow(aAffLc2Im.I10().x*aAffLc2Im.I01().x,2) + pow(aAffLc2Im.I10().y*aAffLc2Im.I01().y,2);
                        //double theta_max =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(0.5);
                        double theta_min =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(-0.5);
                        //cout<<"theta_max : "<<theta_max<<" - theta_min : "<<theta_min<<endl;
                        if (theta_min > min_cur)
                        {
                            min_cur = theta_min;
                            picMaster = apic;
                        }
                    }
                    else
                    {
                        //cout<<" surf :"<<aSurf<<endl;
                    }

                }
                if (picMaster != NULL)
                {
                    //cout<<" ++ min_cur :"<<min_cur<<endl<<" ++ picMaster :"<<picMaster->getNameImgInStr()<<endl;
                    cmptMas[picMaster->mIndex]++;
                }
                else
                    cmptDel++;

            }



            //draw resultat: elipse affinité
            //Pour chaque triangle: creer une cercle se situe dans le plan du triangle
            double aNbPt =20;
            vector<Pt3dr> ptCrlP (aNbPt);
            vector< vector<Pt2dr> > VptCrlI;
            for (uint aKP =0; aKP<aNbPt; aKP++)
                {
                    Pt2dr aPtCrl;
                    aPtCrl = Pt2dr::FromPolar(1, aKP*2*PI/aNbPt);
                    Pt3dr aPtCrlP = aRot_EP.ImAff(Pt3dr(aPtCrl.x, aPtCrl.y, 0));
                    cout<<aPtCrl<<" -> "<<aPtCrlP<<endl;
                    ptCrlP.push_back(aPtCrlP);
                }
            cout<<"Result : "<<endl;
            cout<<" ++Cercle on Plan Triangle :"<<endl;
            for (uint aKP=0; aKP<aNbPt; aKP++)
            {
                cout<<ptCrlP[aKP]<<endl;
            }

/*
            for (uint aKI =0; aKI<PtrPic.size(); aKI++)
                {
                    pic * aPic = PtrPic[aKI];
                    ElAffin2D aAffLc2Im = VaAffLc2Im[aKI];
                    for (uint aKP = 0; aKP<aNbPt; aKP++)
                    {
                        VptCrlI[aKI].push_back(aAffLc2Im(Pt2dr(ptCrlP[aKP].x, ptCrlP[aKP].y)));
                    }
                }



            for (uint aKC=0; aKC<VptCrlI.size(); aKC++)
            {
                cout<<endl<<" ++Img: "<<PtrPic[aKC]->getNameImgInStr()<<endl;
                for (uint aKP=0; aKP<aNbPt; aKP++)
                {
                    cout<<VptCrlI[aKC][aKP]<<endl;
                }
            }

*/

            //Samplizer le cercle (prendre Nb point dans le cercle (from polar(&, aK2pi/Nb))
            //Pour chaque image :
                //calcule coordonne le cercle affinité (coor les point simplizé du cercle )
                //draw seg entre chaque point pour créer une ellipse .


            for (uint acP = 0; acP<PtrPic.size(); acP++)
            {
                cout<<cmptMas[acP]<<endl;
            }
            cout<<"cmptDel "<<cmptDel<<endl;



        }




        return EXIT_SUCCESS;
    }
