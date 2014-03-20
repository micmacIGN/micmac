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


/*******************************************/
/*                                         */
/*    cSolBasculeRig                        */
/*    cBasculementRigide                   */
/*                                         */
/*******************************************/



void PutPt(FILE * aFP,const Pt3dr & aP,bool aModeBin,bool aDouble)
{
    if (aModeBin)
    {
        if (aDouble)
        {
              fwrite(&aP.x,sizeof(aP.x),1,aFP);
              fwrite(&aP.y,sizeof(aP.y),1,aFP);
              fwrite(&aP.z,sizeof(aP.z),1,aFP);
        }
        else
        {
             float x= (float)aP.x;
             float y= (float)aP.y;
             float z= (float)aP.z;
             fwrite(&x,sizeof(float),1,aFP);
             fwrite(&y,sizeof(float),1,aFP);
             fwrite(&z,sizeof(float),1,aFP);
        }
    }
    else
    {
       fprintf(aFP,"%f %f %f ",aP.x,aP.y,aP.z);
    }
}

void  cAppliApero::ClearAllCamPtsVu()
{
    for (int aK=0; aK<int(mVecPose.size()) ; aK++)
    {
         mVecPose[aK]->ResetPtsVu();
    }
}

void cAppliApero::ExportNuage(const cExportNuage & anEN)
{
    const cExportNuageByImage * aByI = anEN.ExportNuageByImage().PtrVal();
    int aNbChan= anEN.NbChan().Val();
    cSetName *  aSelector = mICNM->KeyOrPatSelector(anEN.PatternSel());


     int aNbC = anEN.NbChan().Val();
     std::vector<std::string>  aSet ;
     for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
     {
        const std::string & aName =mVecPose[aKP]->Name();
        if (aSelector->IsSetIn(aName))
        {
             aSet.push_back(aName);
        }
     }
     if ((aNbC==-1) || (aNbC==3))
     {
        MkFMapCmdFileCoul8B(mDC,aSet);
     }

    // cElRegex_Ptr  aSelector = anEN.PatternSel().ValWithDef(0);

    cArgGetPtsTerrain anAGP(1.0,anEN.LimBSurH().Val());
    if (aByI)
    {
       anAGP.SetByIm(true,aByI->SymPts().Val());
       ClearAllCamPtsVu();
    }


    cPonderationPackMesure aPPM = anEN.Pond();
    aPPM.Add2Compens().SetVal(false);
    cStatObs aStatObs(false);

    std::string aImExpoRef="";
    if (anEN.ImExpoRef().IsInit())
    {
       aImExpoRef=mDC+mICNM->Assoc1To1(anEN.KeyFileColImage(),anEN.ImExpoRef().Val(),true);
    }

    eModeAGP aLastMode = eModeAGPNone;

    for 
    (
         std::list<std::string>::const_iterator itL=anEN.NameRefLiaison().begin();
         itL!=anEN.NameRefLiaison().end();
         itL++
    )  
    {
       cPackObsLiaison * aPOL = PackOfInd(*itL);
       std::map<std::string,cObsLiaisonMultiple *> &  aDM = aPOL->DicoMul();
       for
       (
             std::map<std::string,cObsLiaisonMultiple *>::const_iterator  itDM = aDM.begin();
             itDM != aDM.end();
             itDM++
       )
       {
           cObsLiaisonMultiple * anOLM = itDM->second;
           cPoseCam *  aPC =  anOLM->Pose1();
           std::string aNameFile = mICNM->Assoc1To1(anEN.KeyFileColImage(),aPC->Name(),true);
           eModeAGP aMode = eModeAGPIm;

           if (aNameFile == "NoFile")
              aMode = eModeAGPHypso;
           else if (aNameFile == "NormalePoisson")
              aMode = eModeAGPNormale;
           else if (aNameFile == "NoAttr")
              aMode =  eModeAGPNoAttr;
           else if (aNameFile == "NoPoint")
              aMode =  eModeAGPNoPoint;

           if (aLastMode!=eModeAGPNone)
           {
                ELISE_ASSERT(aLastMode==aMode,"Variable mode in cAppliApero::ExportNuage");
           }
           aLastMode = aMode;

           if (aMode==eModeAGPIm)
           {
               aNameFile = mDC+aNameFile;
           }

           if (aSelector->IsSetIn(aPC->Name()) && (aMode != eModeAGPNoPoint))
           {
                if (aMode==eModeAGPIm)
                   anAGP.InitFileColor(aNameFile,-1,aImExpoRef,aNbChan);
                else if (aMode==eModeAGPHypso)
                   anAGP.InitColiorageDirec(anEN.DirCol().Val(),anEN.PerCol().Val());
                else if (aMode==eModeAGPNormale)
                    anAGP.InitModeNormale();
                else if (aMode==eModeAGPNoAttr)
                {
                    anAGP.InitModeNoAttr();
                }

                anOLM->AddObsLM
                (
                     aPPM, 
                     (const cPonderationPackMesure *)0,   // PPM Surf
                     &anAGP,
                     (cArgVerifAero *) 0,
                     aStatObs,
                     (const cRapOnZ *) 0
                );
           }
       }
    }

    if (anEN.NuagePutCam().IsInit())
    {
        cNuagePutCam aNPC = anEN.NuagePutCam().Val();
        double aStIm =  aNPC.StepImage().Val() ;
        for (int aKP=0 ; aKP<int(mVecPose.size()) ; aKP++)
        {
            cPoseCam *  aPC =  mVecPose[aKP]; 
            const CamStenope *  aCS = aPC->CurCam();



            Box2di aBox(Pt2di(0,0),aCS->Sz());
            if (aCS->HasRayonUtile())
            {
                Pt2dr aMil = aCS->Sz() / 2.0;
                Pt2dr aDir = vunit(aMil) * aCS->RayonUtile();
                aBox = Box2di(round_ni(aMil-aDir),round_ni(aMil+aDir));
            }
            Pt2di aTabC[4];
            aBox.Corners(aTabC);
            


            Pt3dr aC3D[4];
            for (int aKC=0 ; aKC<4 ; aKC++)
            {
               aC3D[aKC] = aCS->ImEtProf2Terrain(Pt2dr(aTabC[aKC]),aNPC.Long());
            }

            double aSomD = 0;
            for (int aKC=0 ; aKC<4 ; aKC++)
            {
               aSomD += euclid(aC3D[aKC]-aC3D[(aKC+1)%4]);
            }
            double aProf = aNPC.Long() * ((4*aNPC.Long()) / aSomD);

           
            for (int aKC=0 ; aKC<4 ; aKC++)
            {
               aC3D[aKC] = aCS->ImEtProf2Terrain(Pt2dr(aTabC[aKC]),aProf);
            }

            Pt3dr aCo = aCS->PseudoOpticalCenter();
            for (int aKC=0 ; aKC<4 ; aKC++)
            {
               anAGP.AddSeg(aC3D[aKC],aC3D[(aKC+1)%4],aNPC.StepSeg(),aNPC.ColCadre());
               anAGP.AddSeg(aCo,aC3D[aKC],aNPC.StepSeg(),aNPC.ColRay().ValWithDef(aNPC.ColCadre()));
            }

            {
                std::string   aName = StdPrefixGen(aPC->Name());
                std::string   aNum;
                cElBitmFont & aFont =  cElBitmFont::BasicFont_10x8();
                for (const char * aC=aName.c_str() ; *aC ; aC++)
                {
                    if (isdigit(*aC))
                    {
                        aNum += *aC;
                    }
                }
                if (aNum=="") aNum ="00000";

                const char * aC = aNum.c_str();



                int aSzFX=0;
                int aSzFY=0;
                int aSpace=1;
                while (*aC)
                {
                    Im2D_Bits<1>  anIm = aFont.ImChar(*aC);
                    Pt2di aSz = anIm.sz();
                    aSzFX += aSz.x + aSpace;
                    ElSetMax(aSzFY,aSz.y);
                    aC++;
                }
                Pt2dr aSzC = Pt2dr(aCS->Sz());
           
                int aKC=0;
                double aProp = 0.8;

                double aStep = ElMin(aSzC.x/aSzFX,aSzC.y/aSzFY) * aProp;

                Pt2dr aRab = aSzC - Pt2dr(aSzFX,aSzFY) * aStep;

                aC =  aNum.c_str();
                int    aNb = 3;
                aSzFX = 0;
                while (*aC)
                {
                    Pt3di aCol(255,255,255);
                    Im2D_Bits<1>  anIm = aFont.ImChar(*aC);
                    TIm2DBits<1>  aTIm(anIm);
                    Pt2di aSz = anIm.sz();
                    Pt2di aP;
                    for (aP.x=0 ; aP.x<aSz.x ;aP.x++)
                    {
                        for (aP.y=0 ; aP.y<aSz.y ;aP.y++)
                        {
                            if (aTIm.get(aP))
                            {
                                for (int aKx = 0 ; aKx< aNb ; aKx++)
                                {
                                   for (int aKy = 0 ; aKy< aNb ; aKy++)
                                   {
                                       Pt2di anU = aP + Pt2di(aSzFX,0);
                                       Pt2dr  aPW = Pt2dr(anU.x+aKx/double(aNb),anU.y+aKy/double(aNb)) * aStep ;
                                       aPW = aPW+ aRab/2.0;
                                       Pt3dr aQ =  aCS->NoDistImEtProf2Terrain(aPW,aProf);
                                       anAGP.AddPts(aQ,aCol);
                                   }
                               }
                            }
                        }
                    }
                    aC++;
                    aKC++;
                    aSzFX += aSz.x + aSpace;
                }
            }

            if (aStIm >0)
            {
                std::string aNameFile = mDC+mICNM->Assoc1To1(anEN.KeyFileColImage(),aPC->Name(),true);
                anAGP.InitFileColor(aNameFile,aStIm,aImExpoRef,aNbChan);

                Pt2dr aSZR1 = Pt2dr(aCS->Sz());
                Pt2di aNb = round_up(aSZR1/aStIm); 
                Pt2dr aSt2 = aSZR1.dcbyc(Pt2dr(aNb));
                for (int aX=0 ; aX<=aNb.x ;aX++)
                {
                   for (int aY=0 ; aY<=aNb.y ;aY++)
                   {
                         Pt2dr aPR1(aX*aSt2.x,aY*aSt2.y);
                         anAGP.AddAGP
                         (
                             aPR1,
                             aCS->ImEtProf2Terrain(aPR1,aProf),
                             1,
                             true
                         );
                   }
                }
            }
        }
    }
    bool aModeBin = anEN.PlyModeBin().Val();

    if ((aLastMode==eModeAGPNormale) || (aLastMode==eModeAGPNoAttr))
    {
         FILE * aFP = FopenNN(mDC+anEN.NameOut(),"w","cAppliApero::ExportNuage");


         const std::vector<Pt3dr>  &   aVPts = anAGP.Pts();
         const std::vector<Pt3di>  &   aVNorm = anAGP.Cols();
         if ((aLastMode==eModeAGPNoAttr) && aModeBin)
         {
             int aNb = aVPts.size();
             fwrite(&aNb,sizeof(aNb),1,aFP);
         }


         for (int aK=0; aK<int(aVPts.size()) ; aK++)
         {
              PutPt(aFP,aVPts[aK],aModeBin,aLastMode==eModeAGPNoAttr);
              if (aLastMode==eModeAGPNormale)
              {
                  Pt3dr aNorm = -Pt3dr(aVNorm[aK]) / mAGPFactN;
                  PutPt(aFP,aNorm,aModeBin,false);
              }
              if (! aModeBin) 
                 fprintf(aFP,"\n");
         }

         ElFclose(aFP);
    }
    else
    {
       std::list<std::string> aVCom;
       std::vector<const cElNuage3DMaille *> aVNuage;
       cElNuage3DMaille::PlyPutFile
       (
          mDC+anEN.NameOut(),
          aVCom,
          aVNuage,
          &(anAGP.Pts()),
          &(anAGP.Cols()),
          anEN.PlyModeBin().Val()
       );
    }
    if (aByI)
    {
        for (int aK=0; aK<int(mVecPose.size()) ; aK++)
        {
             cPoseCam & aPC = *(mVecPose[aK]);
             const std::vector<Pt3dr> & aVPts = aPC.PtsVu();
             std::string aName = mICNM->Assoc1To1(aByI->KeyCalc(),aPC.Name(),true);
             FILE * aFP = FopenNN(mDC+aName,"w","cAppliApero::ExportNuage");
             
             if (aModeBin)
             {
                 int aNb = aVPts.size();
                 fwrite(&aNb,sizeof(aNb),1,aFP);
             }
             for (int aK=0; aK<int(aVPts.size()) ; aK++)
             {
                  PutPt(aFP,aVPts[aK],aModeBin,aLastMode==eModeAGPNoAttr);
             }
             ElFclose(aFP);
        }
        ClearAllCamPtsVu();
    }
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
