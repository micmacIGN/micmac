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
#include "general/all.h"

#include "Apero.h"
#include "ext_stl/numeric.h"


/*******************************************/
/*                                         */
/*    cSolBasculeRig                        */
/*    cBasculementRigide                   */
/*                                         */
/*******************************************/

namespace NS_ParamApero
{


void PutPt(FILE * aFP,const Pt3dr & aP,bool aModeBin)
{
    if (aModeBin)
    {
        float x=  aP.x;
        float y=  aP.y;
        float z=  aP.z;
        fwrite(&x,sizeof(float),1,aFP);
        fwrite(&y,sizeof(float),1,aFP);
        fwrite(&z,sizeof(float),1,aFP);
    }
    else
    {
       fprintf(aFP,"%f %f %f ",aP.x,aP.y,aP.z);
    }
}



void cAppliApero::ExportNuage(const cExportNuage & anEN)
{
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

           if (aLastMode!=eModeAGPNone)
           {
                ELISE_ASSERT(aLastMode==aMode,"Variable mode in cAppliApero::ExportNuage");
           }
           aLastMode = aMode;

           if (aMode==eModeAGPIm)
           {
               aNameFile = mDC+aNameFile;
           }

           if (aSelector->IsSetIn(aPC->Name()))
           {
                if (aMode==eModeAGPIm)
                   anAGP.InitFileColor(aNameFile,-1,aImExpoRef,aNbChan);
                else if (aMode==eModeAGPHypso)
                   anAGP.InitColiorageDirec(anEN.DirCol().Val(),anEN.PerCol().Val());
                else if (aMode==eModeAGPNormale)
                    anAGP.InitModeNormale();

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

    if (aLastMode==eModeAGPNormale)
    {
         FILE * aFP = FopenNN(mDC+anEN.NameOut(),"w","cAppliApero::ExportNuage");

         const std::vector<Pt3dr>  &   aVPts = anAGP.Pts();
         const std::vector<Pt3di>  &   aVNorm = anAGP.Cols();
         bool aModeBin = anEN.PlyModeBin().Val();


         for (int aK=0; aK<int(aVPts.size()) ; aK++)
         {
              Pt3dr aNorm = -Pt3dr(aVNorm[aK]) / mAGPFactN;
              PutPt(aFP,aVPts[aK],aModeBin);
              PutPt(aFP,aNorm,aModeBin);
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
          anEN.PlyModeBin().Val(),
          anEN.SavePtsCol().Val()
       );
    }
}


};





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
