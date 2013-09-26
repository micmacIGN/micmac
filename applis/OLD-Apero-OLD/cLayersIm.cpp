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

namespace NS_ParamApero
{

/****************************************************************************/
/*                                                                          */
/*                             cOneImageOfLayer                             */
/*                                                                          */
/****************************************************************************/

const int cOneImageOfLayer::mTheNoLayer = 255;


bool cOneImageOfLayer::LoadFileRed(const std::string & aNameRed)
{
    if ( !ELISE_fp::exist_file(aNameRed))  
       return false;
    mIm = Im2D_U_INT1::FromFileStd(aNameRed);
    mTIm = TIm2D<U_INT1,INT>(mIm);
    Pt2di aSzR = mIm.sz();
    mLabMin = 1000000;
    mLabMax = -1000000;
    Pt2di aP;
    for (aP.x=0 ; aP.x<aSzR.x; aP.x++)
    {
        for (aP.y=0 ; aP.y<aSzR.y; aP.y++)
        {
            int aLab = mTIm.get(aP);
            if (aLab != mTheNoLayer)
            {
               ElSetMin(mLabMin,aLab);
               ElSetMax(mLabMax,aLab);
           }
        }
    }
    std::cout  << "Labels, Min: " << mLabMin << " , Max :" << mLabMax << "\n";
    return true;
}

Im2D_U_INT1 cOneImageOfLayer::MakeImagePrio
     (
         Im2D_U_INT1              aImIn,
         int                      aDeZoom,
         int                      aSzBox
     )
{
// std::cout << "cOneImageOfLayer::MakeImagePrio\n";
    if ((aDeZoom==1) && (aSzBox<=0)) return aImIn;
    TIm2D<U_INT1,INT> aTImIn(aImIn);

    Pt2di aSzOut = aImIn.sz() / aDeZoom;
    Im2D_U_INT1 aImOut(aSzOut.x,aSzOut.y);
    TIm2D<U_INT1,INT> aTImOut(aImOut);

    int pNoDef = mVPrio[mTheNoLayer];

    Pt2di aPBox(aSzBox,aSzBox);
    Pt2di aPOut;

// std::cout << "SZZZ " << aImIn.sz() << aSzOut << "\n";
    for (aPOut.x=0 ; aPOut.x<aSzOut.x; aPOut.x++)
    {
        for (aPOut.y=0 ; aPOut.y<aSzOut.y; aPOut.y++)
        {
            const Pt2di aPIn = aPOut*aDeZoom;
            int aLabC = aTImIn.getproj(aPIn);
            int aPrio = mVPrio[aLabC];
            const Pt2di aP0In = aPIn-aPBox;
            const Pt2di aP1In = aPIn+aPBox;
            Pt2di aQIn;
            for (aQIn.x = aP0In.x ;aQIn.x<=aP1In.x; aQIn.x++)
            {
                  for (aQIn.y = aP0In.y ;aQIn.y<=aP1In.y; aQIn.y++)
                  {
                      int aLab = aTImIn.getproj(aQIn);
                      if (aLab!= aLabC)
                      {
                         ElSetMin(aPrio,pNoDef);
                      }
                      ElSetMin(aPrio,mVPrio[aLab]);
                  }
            }

            aTImOut.oset
            (
                aPOut,
                (aPrio<=pNoDef) ? mVLabOfPrio[aPrio] : aLabC
            );
        }
    }
    return aImOut;
}

cOneImageOfLayer::cOneImageOfLayer
(
      cAppliApero & anAppli,
      const  cLayerImageToPose & aLIP,
      const std::string & aNameIm,
      cOneImageOfLayer *  aLayerTer
)  :
   mAppli       (anAppli), 
   mIm          (1,1),
   mTIm         (mIm),
   mVPrio       (256,255),
   mVLabOfPrio  (256,mTheNoLayer),
   mDeZoom      (aLIP.FactRed()),
   mLayerTer    (aLayerTer),
   mCam         (0),
   mGRRF        (0),
   mSysCam      (0)
{
    const cLayerTerrain * aLT = aLIP.LayerTerrain().PtrVal();
    if (aLT)
    {
       if (mLayerTer) // On dans une image en train de se "brancher" vers le terrain
       {
            mLabMin = mLayerTer->mLabMin;
            mLabMax = mLayerTer->mLabMax;
            std::string aNameCam = mAppli.DC()+ mAppli.ICNM()->Assoc1To1(aLT->KeyAssocOrImage(),aNameIm,true);
            mCam = Cam_Gen_From_File(aNameCam,aLT->TagOri().Val(),mAppli.ICNM());
            return;
       }
       else // On est en train de creer le layer terrain lui meme
       {
           std::string aNameGeoR = mAppli.DC()+ mAppli.ICNM()->Assoc1To1(aLT->KeyAssocGeoref().Val(),aNameIm,true);
           mGRRF = cGeoRefRasterFile::FromFile(aNameGeoR);
           if (aLT->ZMoyen().IsInit())
              mZMoy = aLT->ZMoyen().Val();
           else
              mZMoy = mGRRF->ZMoyen();

           mSysCam = cSysCoord::FromFile(mAppli.DC()+aLT->SysCoIm());
       }
    }
    std::string aNameRed =  mAppli.DC()+ mAppli.ICNM()->Assoc1To1(aLIP.KeyNameRed().Val(),aNameIm,true);
    if (mDeZoom==1)  
    {
        aNameRed =  mAppli.DC()+aNameIm;
    }
    if (LoadFileRed(aNameRed))
       return;

    // ON LOADE l'IMAGE REDUITE ,  INITIALISE les tailles, ALLOUE la memoire

    for (int aK=0 ; aK<int(aLIP.EtiqPrio().size()); aK++) 
    {
       mVPrio[aLIP.EtiqPrio()[aK]] = aK;
    }
    ElSetMin(mVPrio[mTheNoLayer],int(aLIP.EtiqPrio().size()));

    for (int aK=0 ; aK<256; aK++) 
    {
       mVLabOfPrio[mVPrio[aK]] = aK;
    }
    

    Im2D_U_INT1 aImZ1 = Im2D_U_INT1::FromFileStd(anAppli.DC()+aNameIm);
    Im2D_U_INT1 aImTmp = MakeImagePrio(aImZ1,mDeZoom,mDeZoom/2);


    int aFCoh = aLIP.FactCoherence().Val();
    int aSzBox =  (aFCoh>=0) ? ((aFCoh+mDeZoom-1)/mDeZoom) : 0;
    mIm = MakeImagePrio(aImTmp,1,aSzBox);
    mTIm = TIm2D<U_INT1,INT>(mIm);




    Tiff_Im::CreateFromIm(mIm,aNameRed);
    bool OkLoad = LoadFileRed(aNameRed);
    ELISE_ASSERT(OkLoad,"Incoh in cOneImageOfLayer::cOneImageOfLayer");

    std::cout << "   ==== LAB = " << mLabMin << " " << mLabMax << "\n";
}

INT cOneImageOfLayer::LayerOfPt(const Pt2dr & aP)
{
    if (mLayerTer)
    {
       Pt3dr aQ = mCam->F2AndZtoR3(aP,mLayerTer->mZMoy);
       aQ = mLayerTer->mSysCam->ToGeoC(aQ);
       aQ = mLayerTer->mGRRF->Geoc2File(aQ);
// std::cout << aP << aQ << "\n";
       return mLayerTer->LayerOfPt(Pt2dr(aQ.x,aQ.y));
    }
    return mTIm.getproj(round_ni(aP/mDeZoom));
}

void cOneImageOfLayer::SplitLayer
     (
          cOneImageOfLayer& aL2,
          const std::string & aNameH,
          const cSplitLayer & aSL
     )
{
   ElPackHomologue aPck = ElPackHomologue::FromFile(mAppli.DC()+aNameH);
   int aLMin = ElMax(mLabMin,aL2.mLabMin);
   int aLMax = ElMin(mLabMax,aL2.mLabMax);

   for (int aLab = aLMin ; aLab<=aLMax ; aLab++)
   {
       ElPackHomologue aPackR;
       for
       (
            ElPackHomologue::iterator itP=aPck.begin();
            itP!=aPck.end();
            itP++
       )
       {
            if (
                      (LayerOfPt(itP->P1()) == aLab)
                   && (aL2.LayerOfPt(itP->P2()) == aLab)
               )
            {
                aPackR.Cple_Add(ElCplePtsHomologues(itP->P1(),itP->P2()));
            }
       }
 
       std::string aNameRed =  mAppli.DC()+ mAppli.ICNM()->Assoc1To2
                               (
                                    aSL.KeyCalHomSplit(),
                                    aNameH,
                                    ToString(aLab),
                                    true
                               );
       aPackR.StdPutInFile(aNameRed);
       // std::cout << "RRRReedd === " << aNameRed << "\n";
   }

}

/****************************************************************************/
/*                                                                          */
/*                               cLayerImage                                */
/*                                                                          */
/****************************************************************************/

cLayerImage::cLayerImage(cAppliApero & anAppli,const cLayerImageToPose & aLITP) :
    mAppli     (anAppli),
    mLI2P      (aLITP),
    mParamLT   (mLI2P.LayerTerrain().PtrVal()),
    mImTerrain (0)
{
   if ( mParamLT)
   {
       mImTerrain = new cOneImageOfLayer(mAppli,aLITP,mLI2P.KeyCalculImage(),0);
   }
}

bool cLayerImage::IsTerrain() const
{
   return mParamLT != 0;
}

std::string cLayerImage::NamePose2NameLayer(const std::string & aNP)
{
    // En geometrie terrain chaque image genere une entree, pour avoir son orientation
    return   IsTerrain()                                               ?
             aNP                                                       :
             mAppli.ICNM()->Assoc1To1(mLI2P.KeyCalculImage(),aNP,true) ;
}

cOneImageOfLayer * cLayerImage::NamePose2Layer(const std::string & aNP)
{
   // if (IsTerrain())  return mImTerrain;

    std::string aNL = NamePose2NameLayer(aNP);
    cOneImageOfLayer  * aLI = mIms[aNL];
    if (aLI==0)
    {
        std::cout << "For Layer " << aNL << "\n";
        ELISE_ASSERT(false,"Cannot fin layer\n");
    }

    return aLI;
}

void cLayerImage::AddLayer(cPoseCam & aPC)
{
   const std::string & aNC = aPC.Name();
   std::string aNL = NamePose2NameLayer(aNC);
   // std::string aNL = mAppli.ICNM()->Assoc1To1(mLI2P.KeyCalculImage(),aNC,true);
   if (mIms[aNL] ==0)
   {
      std::cout << "CREATE LAYER " << aNL << "\n";
      mIms[aNL] = new cOneImageOfLayer(mAppli,mLI2P,aNL,mImTerrain);
   }
}

void cLayerImage::SplitHomFromImageLayer
     (
            const std::string & aNHom, 
            const cSplitLayer & aSL,
            const std::string & aNIm1, 
            const std::string & aNIm2
     )
{
   cOneImageOfLayer * aLay1 = NamePose2Layer(aNIm1);
   cOneImageOfLayer * aLay2 = NamePose2Layer(aNIm2);

   aLay1->SplitLayer(*aLay2,aNHom,aSL);
}

/****************************************************************************/
/*                                                                          */
/*                               cLayerImage                                */
/*                                                                          */
/****************************************************************************/

cLayerImage * cAppliApero::LayersOfName(const std::string & aName)
{
    cLayerImage * aLI = mMapLayers[aName];
    if (aLI==0)
    {
        std::cout << " Id=" << aName << "\n";
        ELISE_ASSERT(false,"Unkown Layer in cAppliApero::SplitHomFromImageLayer");
    }
    return aLI;
}


void cAppliApero::SplitHomFromImageLayer
     (
            const std::string & aNHom, 
            const cSplitLayer & aSL,
            const std::string & aNIm1, 
            const std::string & aNIm2
     )
{
    cLayerImage * aLI = LayersOfName(aSL.IdLayer());

    aLI->SplitHomFromImageLayer(aNHom,aSL,aNIm1,aNIm2);
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
