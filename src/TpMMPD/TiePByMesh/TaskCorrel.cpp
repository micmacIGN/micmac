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
#include "DrawOnMesh.h"
#include <stdio.h>

const double TT_SEUIL_SURF = 100;
const double TT_SEUIL_RESOLUTION = DBL_MIN;

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
        bool nInteraction = false;
        double aZ = 0.25;
        double aSclElps = -1;
        Pt3dr clIni(255.0,255.0,255.0);
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
                    << EAM(nInteraction, "nInt", true, "nInteraction")
                    << EAM(aZ, "aZ", true, "aZoom image display")
                    << EAM(aSclElps, "aZEl", true, "fix size ellipse display (in pxl)")
                    << EAM(clIni, "clIni", true, "color mesh (=[255,255,255])")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        InitOutil *aChain;
        if (!Test)
        {
            aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
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

            vector< vector<triangle*> > VtriOfImMaster (int(PtrPic.size()));
            vector<AJobCorel> VJobCorrel;
            aChain->getLstJobCorrel(VJobCorrel);
            for (uint aKJob = 0; aKJob<VJobCorrel.size(); aKJob++)
            {
                AJobCorel ajob = VJobCorrel[aKJob];
                VtriOfImMaster[ajob.picM->mIndex].push_back(ajob.tri);
            }

            cout<<endl;
            //export mesh correspond with each image:

            ELISE_fp::MkDirSvp("PLYVerif");
            for (uint akP=0; akP<VtriOfImMaster.size(); akP++)
            {
                DrawOnMesh aDraw;
                std::string filename = aChain->getPrivmICNM()->Dir() + "PLYVerif/" + PtrPic[akP]->getNameImgInStr() + ".ply";
                Pt3dr color(round(akP*255.0/double(VtriOfImMaster.size())), 50, 30);
                aDraw.drawListTriangle(VtriOfImMaster[akP], filename, color);
                cout<<"Draw "<<VtriOfImMaster[akP].size()<<" Tri "<<filename<<endl;
            }
        }
        else
        {
            aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                              SzPtCorr, SzAreaCorr,
                                              corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
            aChain->initAll(pathPlyFileS);
            vector<pic*> PtrPic = aChain->getmPtrListPic();
            vector< vector<triangle*> > VtriOfImMaster (int(PtrPic.size()));
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
            vector<cXml_TriAngulationImMaster> VjobImMaster (int(PtrPic.size()));

            for (uint aKP=0; aKP<PtrPic.size(); aKP++)
            {
                pic * aPic = PtrPic[aKP];
                VjobImMaster[aKP].NameMaster() = aPic->getNameImgInStr();
                for (uint aKP2=0; aKP2<PtrPic.size(); aKP2++)
                {
                    VjobImMaster[aKP].NameSec().push_back(PtrPic[aKP2]->getNameImgInStr());
                }
            }

            ElRotation3D    aRot_PE(ElRotation3D::Id);
            ElRotation3D    aRot_EP(ElRotation3D::Id);
            vector <ElAffin2D> VaAffLc2Im;
            vector<Video_Win * > VVW (PtrPic.size());
            if (nInteraction)
            {
                for (uint aKVW=0; aKVW<PtrPic.size(); aKVW++)
                    VVW[aKVW] = Video_Win::PtrWStd(Pt2di(PtrPic[aKVW]->mImgSz*aZ),true,Pt2dr(aZ,aZ));
            }
            for (uint aIT = 0; aIT<PtrTri.size(); aIT++)
            {

                cXml_Triangle3DForTieP jobATri;
                triangle * atri = PtrTri[aIT];
                cout<<"+ TRI : "<<aIT<<endl;
                min_cur = DBL_MIN;
                picMaster = NULL;
                jobATri.P1() = atri->getSommet(0);
                jobATri.P2() = atri->getSommet(1);
                jobATri.P3() = atri->getSommet(2);
                vector<pic*> VPicAcpt;
                vector<vector<Pt2dr> >VVCl (PtrPic.size());
                for (uint aIP = 0; aIP<PtrPic.size(); aIP++)
                {
                    pic * apic = PtrPic[aIP];
                    Tri2d aTriIm = *atri->getReprSurImg()[apic->mIndex];
                    Pt2dr aPtI0 = aTriIm.sommet1[0];    //sommet triangle on img
                    Pt2dr aPtI1 = aTriIm.sommet1[1];
                    Pt2dr aPtI2 = aTriIm.sommet1[2];
                    double aSurf =  (aPtI0-aPtI1) ^ (aPtI0-aPtI2);
                    if (ElAbs(aSurf) > TT_SEUIL_SURF && aTriIm.insidePic)
                    {
                        //creer plan 3D local contient triangle
                        cElPlan3D * aPlanLoc = new cElPlan3D(atri->getSommet(0) , atri->getSommet(1), atri->getSommet(2));

                        aRot_PE = aPlanLoc->CoordPlan2Euclid();
                        aRot_EP = aRot_PE.inv();
                        //calcul coordonne sommet triangle dans plan 3D local (devrait avoir meme Z)
                        Pt3dr aPtP0 = aRot_EP.ImAff(atri->getSommet(0)); //sommet triangle on plan local
                        Pt3dr aPtP1 = aRot_EP.ImAff(atri->getSommet(1));
                        Pt3dr aPtP2 = aRot_EP.ImAff(atri->getSommet(2));





                        //creer translation entre coordonne image global -> coordonne image local du triangle (plan image)
                        ElAffin2D aAffImG2ImL(ElAffin2D::trans(aPtI0));
                        Pt2dr aPtPIm0 = aAffImG2ImL(aPtI0);
                        Pt2dr aPtPIm1 = aAffImG2ImL(aPtI1);
                        Pt2dr aPtPIm2 = aAffImG2ImL(aPtI2);

/*
                        //creer plan 2D local dans plan image (deplacer origin haut gauche d'image -> origin avec 1 sommet de triangle)
                        cElPlan3D * aPlanIm = new cElPlan3D (   Pt3dr(aPtI0.x, aPtI0.y, 0),
                                                                Pt3dr(aPtI1.x, aPtI1.y, 0),
                                                                Pt3dr(aPtI2.x, aPtI2.y, 0)   );

                        ElRotation3D aRot_EPIm = aPlanIm->CoordPlan2Euclid().inv();
                        //calcul coordonne sommet triangle dans plan 2D local
                        Pt3dr aPtPIm0 = aRot_EPIm.ImAff(Pt3dr(aPtI0.x, aPtI0.y, 0));
                        Pt3dr aPtPIm1 = aRot_EPIm.ImAff(Pt3dr(aPtI1.x, aPtI1.y, 0));
                        Pt3dr aPtPIm2 = aRot_EPIm.ImAff(Pt3dr(aPtI2.x, aPtI2.y, 0) );

                        cout<<"aPtPIm0 "<<aPtPIm0<<aPtPIm1<<aPtPIm2<<endl;
*/

                        //calcul affine entre plan 3D local (elimine Z) et plan 2D local
                        ElAffin2D aAffLc2Im;
                        aAffLc2Im = aAffLc2Im.FromTri2Tri(  Pt2dr(aPtP0.x, aPtP0.y),
                                                            Pt2dr(aPtP1.x, aPtP1.y),
                                                            Pt2dr(aPtP2.x, aPtP2.y),
                                                            aPtPIm0,aPtPIm1,aPtPIm2
                                                         );
                        VaAffLc2Im.push_back(aAffLc2Im);

                        //calcul le cercle discretize dans le plan 3D local
                        double rho;
                        if (aSclElps == -1)
                        {
                            double rho1 = sqrt(aPtP1.x*aPtP1.x + aPtP1.y*aPtP1.y);
                            double rho2 = sqrt(aPtP2.x*aPtP2.x + aPtP2.y*aPtP2.y);
                            if (rho1 > rho2)
                                rho = rho1;
                            else
                                rho = rho2;

                        }
                        else
                        {
                            double scale = euclid ( aAffLc2Im.inv()(Pt2dr(0,0)) - aAffLc2Im.inv()(Pt2dr(aSclElps,0)) );
                            rho = scale;
                        }
                        double aNbPt = 100;
                        vector<Pt2dr> VCl;
                        for (uint aKP=0; aKP<aNbPt; aKP++)
                        {
                            Pt2dr ptCrlImg;
                            ptCrlImg = aAffLc2Im(Pt2dr::FromPolar(rho, aKP*2*PI/aNbPt));
                            VCl.push_back(ptCrlImg);
                        }
                        VVCl[aIP] = VCl;

                        //calcul vector max min pour choisir img master
                        double vecA_cr =  aAffLc2Im.I10().x*aAffLc2Im.I10().x + aAffLc2Im.I10().y*aAffLc2Im.I10().y;
                        double vecB_cr =  aAffLc2Im.I01().x*aAffLc2Im.I01().x + aAffLc2Im.I01().y*aAffLc2Im.I01().y;
                        double AB_cr   =  pow(aAffLc2Im.I10().x*aAffLc2Im.I01().x,2) + pow(aAffLc2Im.I10().y*aAffLc2Im.I01().y,2);
                        //double theta_max =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(0.5);
                        double theta_min =  vecA_cr + vecB_cr +sqrt((vecA_cr - vecB_cr) + 4*AB_cr)*(-0.5);
                        cout<<" ++ theta_min : "<<theta_min<<" - "<<apic->getNameImgInStr()<<endl;
                        if (theta_min > min_cur)
                        {
                            min_cur = theta_min;
                            picMaster = apic;
                        }
                        if (theta_min > TT_SEUIL_RESOLUTION)
                            VPicAcpt.push_back(apic);
                    }
                    else
                    {
                        cout<<" ++surf :"<<aSurf<<endl;
                    }

                }
                if (picMaster != NULL && VPicAcpt.size() > 1)
                {
                    //cout<<" ++ min_cur :"<<min_cur<<endl<<" ++ picMaster :"<<picMaster->getNameImgInStr()<<endl;
                    cmptMas[picMaster->mIndex]++;
                    for (uint aKP=0; aKP<VPicAcpt.size(); aKP++)
                    {
                        if (VPicAcpt[aKP]->mIndex != picMaster->mIndex)
                        {
                            jobATri.NumImSec().push_back(VPicAcpt[aKP]->mIndex);
                        }
                    }
                    VjobImMaster[picMaster->mIndex].Tri().push_back(jobATri);
                    VtriOfImMaster[picMaster->mIndex].push_back(atri);
                    cout<<" +Mas : "<<picMaster->getNameImgInStr()<<endl;
                    //display ellipse
                    if (nInteraction)
                    {
                        for (uint aKC=0; aKC<VVCl.size(); aKC++)
                        {
                            Video_Win * aVW = VVW[aKC];
                            Disc_Pal Pdisc = Disc_Pal::P8COL();
                            Gray_Pal Pgr (30);
                            Circ_Pal Pcirc = Circ_Pal::PCIRC6(30);
                            RGB_Pal Prgb (255,1,1);
                            Elise_Set_Of_Palette SOP(NewLElPal(Pdisc) +Elise_Palette(Pgr)+Elise_Palette(Prgb)+Elise_Palette(Pcirc));
                            Line_St lstLineG(Pdisc(P8COL::green),1);
                            pic * aPic = PtrPic[aKC];
                            aVW->set_sop(SOP);
                            aVW->set_title(aPic->getNameImgInStr().c_str());
                            ELISE_COPY(aVW->all_pts(), aPic->mPic_Im2D->in_proj() ,aVW->ogray());
                            aVW->draw_poly_ferm(VVCl[aKC], lstLineG);
                            if (aKC == VVCl.size()-1)
                                aVW->clik_in();
                        }
                    }
                }
                else
                    cmptDel++;

            }
            aDirXML = aDirXML + "Test";
            ELISE_fp::MkDirSvp(aDirXML);
            for (uint aKP=0; aKP<VjobImMaster.size(); aKP++)
            {
                string fileXML =  aChain->getPrivmICNM()->Dir() + aDirXML + "/" + VjobImMaster[aKP].NameMaster() + ".xml";
                MakeFileXML(VjobImMaster[aKP], fileXML);
            }
            cout<<endl;

            for (uint acP = 0; acP<PtrPic.size(); acP++)
            {
                cout<<cmptMas[acP]<<endl;
            }
            cout<<"cmptDel "<<cmptDel<<endl;
            //export mesh correspond with each image:
            ELISE_fp::MkDirSvp("PLYVerif");
            for (uint akP=0; akP<VtriOfImMaster.size(); akP++)
            {
                DrawOnMesh aDraw;
                std::string filename = aChain->getPrivmICNM()->Dir() + "PLYVerif/" + PtrPic[akP]->getNameImgInStr() + ".ply";
                Pt3dr color(round(akP*clIni.x/double(VtriOfImMaster.size())), round(akP*clIni.y/double(VtriOfImMaster.size())), round(akP*clIni.z/double(VtriOfImMaster.size())));
                aDraw.drawListTriangle(VtriOfImMaster[akP], filename, color);
                cout<<"Draw "<<VtriOfImMaster[akP].size()<<" Tri "<<filename<<endl;
            }
        }






        return EXIT_SUCCESS;
    }
