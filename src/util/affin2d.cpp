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


  // Test de Greg

#include "StdAfx.h"

/*
void XXXXX(FILE * aF)
{
  int x;
  int TOTO;
  TOTO = fscanf(aF,"%d",&x);
}
*/




ElAffin2D::ElAffin2D
(
     Pt2dr im00,  // partie affine
     Pt2dr im10,  // partie vecto
     Pt2dr im01
) :
    mI00 (im00),
    mI10 (im10),
    mI01 (im01)
{
}


ElAffin2D::ElAffin2D() :
    mI00 (0,0),
    mI10 (1,0),
    mI01 (0,1)
{
}

bool ElAffin2D::IsId() const
{
   return 
           (mI00==Pt2dr(0,0))
        && (mI10==Pt2dr(1,0))
        && (mI01==Pt2dr(0,1)) ;
}

ElAffin2D ElAffin2D::Id()
{
   return ElAffin2D();
}

ElAffin2D ElAffin2D::trans(Pt2dr aTr)
{
   return ElAffin2D(aTr,Pt2dr(1,0),Pt2dr(0,1));
}





ElAffin2D::ElAffin2D (const ElSimilitude & aSim) :
    mI00 (aSim(Pt2dr(0,0))),
    mI10 (aSim(Pt2dr(1,0)) -mI00),
    mI01 (aSim(Pt2dr(0,1)) -mI00)
{
}

ElAffin2D ElAffin2D::operator * (const ElAffin2D & sim2) const 
{
    return ElAffin2D
           (
              (*this)(sim2(Pt2dr(0,0))),
              IVect(sim2.IVect(Pt2dr(1,0))),
              IVect(sim2.IVect(Pt2dr(0,1)))
           );

}
ElAffin2D ElAffin2D::operator + (const ElAffin2D & sim2) const 
{
    return ElAffin2D
           (
               mI00 + sim2.mI00,
               mI10 + sim2.mI10,
               mI01 + sim2.mI01
           );

}

ElAffin2D ElAffin2D::CorrectWithMatch(Pt2dr aPt,Pt2dr aRes) const
{
    Pt2dr aGot = (*this) (aPt);

    return ElAffin2D
           (
               mI00 + aRes-aGot,
               mI10,
               mI01
           );
}


ElAffin2D ElAffin2D::inv () const
{
    REAL delta = mI10 ^ mI01;

    Pt2dr  Inv10 = Pt2dr(mI01.y,-mI10.y) /delta;
    Pt2dr  Inv01 = Pt2dr(-mI01.x,mI10.x) /delta;

    return  ElAffin2D
            (
                 -(Inv10*mI00.x+Inv01*mI00.y),
                 Inv10,
                 Inv01
            );
}

ElAffin2D ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,Pt2dr aResol,Pt2dr * aSzInOut)
{
   ElAffin2D aRes
             (
                   -Pt2dr(aTr.x/aResol.x,aTr.y/aResol.y),
                   Pt2dr(1.0/aResol.x,0.0),
                   Pt2dr(0.0,1.0/aResol.y)
             );

   if (aSzInOut)
   {
      Box2dr aBoxIn(aTr, aTr+*aSzInOut);
      Box2dr aBoxOut  = aBoxIn.BoxImage(aRes);

      *aSzInOut = aBoxOut.sz();
       aRes = trans(-aBoxOut._p0) * aRes;
   }

   return aRes;
}

ElAffin2D  ElAffin2D::TransfoImCropAndSousEch(Pt2dr aTr,double aResol,Pt2dr * aSzInOut)
{
   return TransfoImCropAndSousEch(aTr,Pt2dr(aResol,aResol),aSzInOut);
}


ElAffin2D  ElAffin2D::L2Fit(const  ElPackHomologue & aPack,double *aResidu)
{
   ELISE_ASSERT(aPack.size()>=3,"Less than 3 point in ElAffin2D::L2Fit");

   static L2SysSurResol aSys(6);
   aSys.GSSR_Reset(false);


   //   C0 X1 + C1 Y1 +C2 =  X2     (C0 C1)  (X1)   C2
   //                               (     )  (  ) +
   //   C3 X1 + C4 Y1 +C5 =  Y2     (C3 C4)  (Y1)   C5

  double aCoeffX[6]={1,1,1,0,0,0};
  double aCoeffY[6]={0,0,0,1,1,1};


   for 
   (
        ElPackHomologue::const_iterator it=aPack.begin();
        it!=aPack.end();
        it++
   )
   {
       aCoeffX[0] = it->P1().x;
       aCoeffX[1] = it->P1().y;

       aCoeffY[3] = it->P1().x;
       aCoeffY[4] = it->P1().y;

       aSys.AddEquation(1,aCoeffX, it->P2().x);
       aSys.AddEquation(1,aCoeffY, it->P2().y);
   }

   Im1D_REAL8 aSol = aSys.Solve(0);
   double * aDS = aSol.data();

   Pt2dr aIm00(aDS[2],aDS[5]);
   Pt2dr aIm10(aDS[0],aDS[3]);
   Pt2dr aIm01(aDS[1],aDS[4]);


   ElAffin2D aRes(aIm00,aIm10,aIm01);

   if (aResidu)
   {
      *aResidu = 0;
      for 
      (
           ElPackHomologue::const_iterator it=aPack.begin();
           it!=aPack.end();
           it++
      )
      {
          *aResidu +=  euclid(aRes(it->P1()),it->P2()) ;
      }
      int aNbPt = aPack.size();
      if (aNbPt>3)
          *aResidu /= (aNbPt-3);
   }
   return aRes;
}


ElAffin2D ElAffin2D::FromTri2Tri
          (
               const Pt2dr & a0, const Pt2dr & a1, const Pt2dr & a2,
               const Pt2dr & b0, const Pt2dr & b1, const Pt2dr & b2
          )
{
     ElAffin2D aA(a0,a1-a0,a2-a0);
     ElAffin2D aB(b0,b1-b0,b2-b0);

     return aB * aA.inv();
}

cElHomographie ElAffin2D::ToHomographie() const
{
    cElComposHomographie aHX(mI10.x,mI01.x,mI00.x);
    cElComposHomographie aHY(mI10.y,mI01.y,mI00.y);
    cElComposHomographie aHZ(     0,     0,     1);

    return  cElHomographie(aHX,aHY,aHZ);
}

// -------------------- :: -------------------
cXml_Map2D MapFromElem(const cXml_Map2DElem & aMapE)
{
    cXml_Map2D aRes;
    aRes.Maps().push_back(aMapE);
    return aRes;
}

//--------------------------------------------


cElMap2D *  ElAffin2D::Map2DInverse() const
{
   return  new ElAffin2D(inv());
}

cXml_Map2D ElAffin2D::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Aff().SetVal(El2Xml(*this));
   return cXml_Map2D(MapFromElem(anElem));
}



/*****************************************************/
/*                                                   */
/*            ElSimilitude                           */
/*                                                   */
/*****************************************************/

cElMap2D * ElSimilitude::Map2DInverse() const
{
    return new ElSimilitude(inv());
}

cXml_Map2D  ElSimilitude::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Sim().SetVal(El2Xml(*this));
   return cXml_Map2D(MapFromElem(anElem));
}


/*****************************************************/
/*                                                   */
/*            cCamAsMap                              */
/*                                                   */
/*****************************************************/

cCamAsMap::cCamAsMap(CamStenope * aCam,bool aDirect)  :
     mCam   (aCam),
     mDirect (aDirect)
{
}

Pt2dr cCamAsMap::operator () (const Pt2dr & p) const
{
   return  mDirect  ? 
           mCam->DistDirecte(p) :  
           mCam->DistInverse(p);
}

cElMap2D * cCamAsMap::Map2DInverse() const
{
   return new cCamAsMap(mCam,!mDirect);
}

cXml_Map2D    cCamAsMap::ToXmlGen()
{
   cXml_MapCam aXmlCam;

   aXmlCam.Directe() = mDirect;
   aXmlCam.PartieCam() = mCam->ExportCalibInterne2XmlStruct(mCam->Sz());

   cXml_Map2DElem anElem;
   anElem.Cam().SetVal(aXmlCam);
   return cXml_Map2D(MapFromElem(anElem));
}


/*****************************************************/
/*                                                   */
/*            cElHomographie                         */
/*                                                   */
/*****************************************************/


cElMap2D * cElHomographie::Map2DInverse() const
{
    return new cElHomographie(Inverse());
}

cXml_Map2D   cElHomographie::ToXmlGen()
{
   cXml_Map2DElem anElem;
   anElem.Homog().SetVal(ToXml());
   return cXml_Map2D(MapFromElem(anElem));
}

Pt2dr  cElHomographie::operator() (const Pt2dr & aP) const
{
   return Direct(aP);
}

/*****************************************************/
/*                                                   */
/*            cElMap2D                               */
/*                                                   */
/*****************************************************/

cElMap2D * cElMap2D::Map2DInverse() const
{
   ELISE_ASSERT(false,"No def cElMap2D::Map2DInverse");
   return 0;
}

cElMap2D * cElMap2D::Simplify() 
{
   return this;
}

cXml_Map2D      cElMap2D::ToXmlGen()
{
   ELISE_ASSERT(false,"No def cElMap2D::ToXmlGen");
   return cXml_Map2D();
}

void   cElMap2D::SaveInFile(const std::string & aName)
{
    cXml_Map2D aXml = ToXmlGen();
    MakeFileXML(aXml,aName);
}

cElMap2D *  Map2DFromElem(const cXml_Map2DElem & aXml)
{
   if (aXml.Homog().IsInit()) return new cElHomographie(aXml.Homog().Val());
   if (aXml.Sim().IsInit()) return new ElSimilitude(Xml2EL(aXml.Sim().Val()));
   if (aXml.Aff().IsInit()) return new ElAffin2D(Xml2EL(aXml.Aff().Val()));
   if (aXml.Cam().IsInit())
   {
       CamStenope* aCS = Std_Cal_From_CIC(aXml.Cam().Val().PartieCam());
       return new cCamAsMap(aCS,aXml.Cam().Val().Directe());
   }


   ELISE_ASSERT(false,"Map2DFromElem");
   return 0;
}

cElMap2D *  cElMap2D::FromFile(const std::string & aName)
{
   cXml_Map2D aXml = StdGetFromSI(aName,Xml_Map2D);
   std::vector<cElMap2D *> aVMap;

   for (std::list<cXml_Map2DElem>::const_iterator itM=aXml.Maps().begin() ; itM!=aXml.Maps().end() ; itM++)
   {
      aVMap.push_back(Map2DFromElem(*itM));
   }


   return new cComposElMap2D(aVMap);
}

/*****************************************************/
/*                                                   */
/*            cElMap2D                               */
/*                                                   */
/*****************************************************/

cComposElMap2D::cComposElMap2D(const std::vector<cElMap2D *>  & aVMap) :
   mVMap (aVMap)
{
}

Pt2dr cComposElMap2D::operator () (const Pt2dr & aP)  const
{
   Pt2dr aRes = aP;
   for (int aK=0 ; aK<int(mVMap.size()) ; aK++)
       aRes = (*(mVMap[aK]))(aRes);
   return aRes;
}

cElMap2D *  cComposElMap2D::Map2DInverse() const
{
   std::vector<cElMap2D *> aVInv;
   for (int aK=int(mVMap.size()-1) ; aK>=0 ; aK--)
      aVInv.push_back(mVMap[aK]->Map2DInverse());

   return new cComposElMap2D(aVInv);
}

cElMap2D * cComposElMap2D::Simplify() 
{
   if (mVMap.size()==1) 
      return mVMap[0];

   return this;
}


cXml_Map2D    cComposElMap2D::ToXmlGen()
{
   cXml_Map2D aRes;

   for (int aK=0 ; aK<int(mVMap.size()) ; aK++)
   {
        cXml_Map2D aXml = mVMap[aK]->ToXmlGen();
        for (std::list<cXml_Map2DElem>::const_iterator itM2=aXml.Maps().begin() ; itM2!=aXml.Maps().end() ; itM2++)
        {
            aRes.Maps().push_back(*itM2);
        }
   }
   return aRes;
}



int CPP_CalcMapHomogr(int argc,char** argv)
{
    std::string aName1,aName2,aNameOut,aSH,anExt="dat";
    std::string anOri;
    int NbTest =50;
    double  Perc = 80.0;
    int     NbMaxPts= 10000;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aName1,"Name Im1")
                    <<  EAMC(aName2,"Name Im2")
                    <<  EAMC(aNameOut,"Name Out"),
        LArgMain()  <<  EAM(aSH,"SH",true,"Set of homologue")
                    <<  EAM(anOri,"Ori",true,"Directory to read distorsion")
    );

    cElemAppliSetFile anEASF(aName1);
    cInterfChantierNameManipulateur * anICNM = anEASF.mICNM;
    std::string aDir = anEASF.mDir;

   


    CamStenope * aCS1=0,*aCS2=0;
    if (EAMIsInit(&anOri))
    {
         StdCorrecNameOrient(anOri,aDir);
         aCS1 = anICNM->GlobCalibOfName(aName1,anOri,false);
         aCS2 = anICNM->GlobCalibOfName(aName2,anOri,false);

         aCS1->Get_dist().SetCameraOwner(aCS1);
         aCS2->Get_dist().SetCameraOwner(aCS2);
    }

    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
                       +  std::string(aSH)
                       +  std::string("@")
                       +  std::string(anExt);


    std::string aNameIn = aDir + anICNM->Assoc1To2(aKHIn,aName1,aName2,true);
    ElPackHomologue aPackIn =  ElPackHomologue::FromFile(aNameIn);
    ElPackHomologue aPackInitial = aPackIn; 

    if (aCS1)
    {
        for (ElPackHomologue::iterator itCpl=aPackIn.begin();itCpl!=aPackIn.end() ; itCpl++)
        {
            itCpl->P1() = aCS1->DistInverse(itCpl->P1());
            itCpl->P2() = aCS2->DistInverse(itCpl->P2());
        }
    }


    double anEcart,aQuality;
    bool Ok;
    cElHomographie aHom = cElHomographie::RobustInit(anEcart,&aQuality,aPackIn,Ok,NbTest,Perc,NbMaxPts);


    std::vector<cElMap2D *> aVMap;
    if (aCS1)
    {
       aVMap.push_back(new cCamAsMap(aCS1,false));
    }
    aVMap.push_back(&aHom);
    if (aCS2)
    {
       aVMap.push_back(new cCamAsMap(aCS2,true));
    }

    cComposElMap2D aComp(aVMap);


    for (ElPackHomologue::iterator itCpl=aPackInitial.begin();itCpl!=aPackInitial.end() ; itCpl++)
    {
        double aD = euclid(aComp(itCpl->P1())-itCpl->P2());
        double aD2 = euclid(aComp(itCpl->P2())-itCpl->P1());
        std::cout << "DIST " << aD << " " << aD2 << "\n";
    }
    MakeFileXML(aComp.ToXmlGen(),aNameOut);


    std::cout << "PACKK = " << aPackIn.size() << "\n";



    return EXIT_SUCCESS;

}


int CPP_ReechImMap(int argc,char** argv)
{
    std::string aNameIm,aNameMap;
    Pt2di aSzOut;
    std::string aNameOut;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  <<  EAMC(aNameIm,"Name Im1")
                    <<  EAMC(aNameMap,"Name map"),
        LArgMain()  
    );

    if (!EAMIsInit(&aNameOut))
       aNameOut = DirOfFile(aNameIm) + "Reech_" + NameWithoutDir(StdPrefix(aNameIm)) +".tif";

    cElMap2D * aMap = cElMap2D::FromFile(aNameMap);

    Tiff_Im aTifIn = Tiff_Im::StdConvGen(aNameIm,-1,true);


    std::vector<Im2DGen *>  aVecImIn =  aTifIn.ReadVecOfIm();
    int aNbC = aVecImIn.size();
    Pt2di aSzIn = aVecImIn[0]->sz();
    if (! EAMIsInit(&aSzOut))
       aSzOut = aSzIn;

    std::vector<Im2DGen *> aVecImOut =  aTifIn.VecOfIm(aSzOut);

    std::vector<cIm2DInter*> aVInter;
    for (int aK=0 ; aK<aNbC ; aK++)
    {
        aVInter.push_back(aVecImIn[aK]->SinusCard(5,5));
    }


    Pt2di aP;
    for (aP.x =0 ; aP.x<aSzOut.x ; aP.x++)
    {
        for (aP.y =0 ; aP.y<aSzOut.y ; aP.y++)
        {
            Pt2dr aQ = (*aMap)(Pt2dr(aP));
            for (int aK=0 ; aK<aNbC ; aK++)
            {
                double aV = aVInter[aK]->GetDef(aQ,0);
                aVecImOut[aK]->SetR(aP,aV);
            }
        }
    }

    Tiff_Im aTifOut
            (
                aNameOut.c_str(),
                aSzOut,
                aTifIn.type_el(),
                Tiff_Im::No_Compr,
                aTifIn.phot_interp()
            );

    ELISE_COPY(aTifOut.all_pts(),StdInPut(aVecImOut),aTifOut.out());

    return EXIT_SUCCESS;
}
  
         // static cElMap2D * FromFile(const std::string &);
         // virtual cXml_Map2D *     ToXmlGen() ; // P

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
