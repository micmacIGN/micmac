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
#include "ext_stl/numeric.h"



/****************************************************************/
/*                                                              */
/*                 cElComposHomographie                         */
/*                                                              */
/****************************************************************/

cElComposHomographie::cElComposHomographie(REAL aX,REAL aY,REAL a1) :
   mX (aX),
   mY (aY),
   m1 (a1)
{
}

cElComposHomographie::cElComposHomographie(const cXmlAffinR2ToR & anAff) :
    mX (anAff.CoeffX()),
    mY (anAff.CoeffY()),
    m1 (anAff.Coeff1())
{
}

cXmlAffinR2ToR cElComposHomographie::ToXml() const
{
   cXmlAffinR2ToR aRes;
   aRes.CoeffX() = mX;
   aRes.CoeffY() = mY;
   aRes.Coeff1() = m1;

   return aRes;
}

bool  cElComposHomographie::HasNan() const
{
    return std_isnan(mX) || std_isnan(mY) || std_isnan(m1);
}

void cElComposHomographie::Show(const std::string & aMes)
{
    std::cout << aMes << " " << mX << " " << mY << " " << m1 << "\n";
}

void cElComposHomographie::Divide(REAL aFact)
{
    mX /= aFact;
    mY /= aFact;
    m1 /= aFact;
}

cElComposHomographie cElComposHomographie::MulXY(REAL aLambda) const
{
    return cElComposHomographie(mX*aLambda,mY*aLambda,m1);
}

cElComposHomographie cElComposHomographie::MulCste(REAL aLambda) const
{
   return cElComposHomographie(mX,mY,m1*aLambda);
}

Fonc_Num cElComposHomographie::operator() (Pt2d<Fonc_Num> aP) const
{
   return mX*aP.x + mY*aP.y + m1;
}

void cElComposHomographie::write(ELISE_fp & aFile) const
{
   aFile.write(mX);
   aFile.write(mY);
   aFile.write(m1);
}

cElComposHomographie cElComposHomographie::read(ELISE_fp & aFile)
{
    REAL aX = aFile.read(&aX);
    REAL aY = aFile.read(&aY);
    REAL a1 = aFile.read(&a1);

    return cElComposHomographie(aX,aY,a1);
}

REAL & cElComposHomographie::CoeffX() {return mX;}
REAL & cElComposHomographie::CoeffY() {return mY;}
REAL & cElComposHomographie::Coeff1() {return m1;}

REAL  cElComposHomographie::CoeffX() const  {return mX;}
REAL  cElComposHomographie::CoeffY() const  {return mY;}
REAL  cElComposHomographie::Coeff1() const  {return m1;}

void cElComposHomographie::SetCoHom(REAL * aData) const
{
   aData[0] = mX;
   aData[1] = mY;
   aData[2] = m1;
}




/****************************************************************/
/*                                                              */
/*                     cElHomographie                           */
/*                                                              */
/****************************************************************/

static void AddCoeffSysHom(cGenSysSurResol * aSys,Pt2dr aP1,Pt2dr aP2,REAL aPds,bool inX)
{
    double Coeff[8] = {0,0,0,0,  0,0,0,0};
    int INumerator = inX ? 0 : 3;

    REAL aC2 = inX ? aP2.x : aP2.y;
    Coeff[INumerator] =   aP1.x;
    Coeff[INumerator+1] = aP1.y;
    Coeff[INumerator+2] = 1;
    Coeff[6] = -aC2 * aP1.x;
    Coeff[7] = -aC2 * aP1.y;

    aSys->GSSR_AddNewEquation(aPds,Coeff,aC2,0);
}

static cElComposHomographie   ComputeHomFromHomologues
                              (
                                  const ElPackHomologue & aPack,
                                  bool aL2,
                                  cElComposHomographie& aHX,
                                  cElComposHomographie& aHY,
                                  cElComposHomographie& aHZ
                               )
{
     ElPackHomologue::const_iterator anIt=aPack.begin();
     // 0 Point : identite 
     if (aPack.size() ==0)
     {
        aHX = cElComposHomographie(1,0,0);
        aHY = cElComposHomographie(0,1,0);
        aHZ = cElComposHomographie(0,0,1);

        return  aHX;
     }


 // 1 Point : translation 
     Pt2dr P1 =  anIt->P1();
     Pt2dr P2 =  anIt->P2();
     anIt++;

     if (aPack.size() ==1)
     {
        Pt2dr aTr = P2 - P1;
        aHX = cElComposHomographie(1,0,aTr.x);
        aHY = cElComposHomographie(0,1,aTr.y);
        aHZ = cElComposHomographie(0,0,1);

        return  aHX;
     }

 // 2 Point : Similitude 
     Pt2dr Q1 =  anIt->P1();
     Pt2dr Q2 =  anIt->P2();
     anIt++;

     if (aPack.size() ==2)
     {

        Pt2dr W = (Q2-P2) / (Q1-P1);
        Pt2dr aTr = P2 - W* P1;

        aHX = cElComposHomographie(W.x,-W.y,aTr.x);
        aHY = cElComposHomographie(W.y,W.x,aTr.y);
        aHZ = cElComposHomographie(0,0,1);

        return  aHX;
     }

 // 3 Point : Affinite 
     Pt2dr R1 =  anIt->P1();
     Pt2dr R2 =  anIt->P2();
     anIt++;

     if (aPack.size() ==3)
     {
        ElMatrix<REAL> M1(2,2); SetCol(M1,0,Q1-P1); SetCol(M1,1,R1-P1);
        ElMatrix<REAL> M2(2,2); SetCol(M2,0,Q2-P2); SetCol(M2,1,R2-P2);

        ElMatrix<REAL> M = M2 * gaussj(M1);

        Pt2dr aTr = P2 - M* P1;

        aHX = cElComposHomographie(M(0,0),M(1,0),aTr.x);
        aHY = cElComposHomographie(M(0,1),M(1,1),aTr.y);
        aHZ = cElComposHomographie(0,0,1);

        return  aHX;
     }








     cGenSysSurResol * aSys = 
	aL2 ? 
	( cGenSysSurResol *) new L2SysSurResol(8) : 
	( cGenSysSurResol *) new SystLinSurResolu(8,aPack.size());

    aSys->SetPhaseEquation(0);
     
     for ( anIt=aPack.begin() ; anIt !=aPack.end(); anIt++)
     {
         AddCoeffSysHom(aSys,anIt->P1(),anIt->P2(),anIt->Pds(),true );
         AddCoeffSysHom(aSys,anIt->P1(),anIt->P2(),anIt->Pds(),false);
     }

    Im1D_REAL8 aSol = aSys->GSSR_Solve((bool *)0);
    REAL * aS = aSol.data();

    aHX = cElComposHomographie(aS[0],aS[1],aS[2]);
    aHY = cElComposHomographie(aS[3],aS[4],aS[5]);
    aHZ = cElComposHomographie(aS[6],aS[7],   1 );

    delete aSys;
    return aHX;
}

void cElHomographie::Normalize()
{
    REAL aFact = mHZ.m1;
    mHX.Divide(aFact);
    mHY.Divide(aFact);
    mHZ.Divide(aFact);
}

void cElHomographie::Show()
{
    mHX.Show("X");
    mHY.Show("Y");
    mHZ.Show("Z");
}

bool  cElHomographie::HasNan() const
{
   return mHX.HasNan() || mHY.HasNan() ||   mHZ.HasNan();
}

cElHomographie::cElHomographie
(
    const cElComposHomographie & aHX,
    const cElComposHomographie & aHY,
    const cElComposHomographie & aHZ
)  :
   mHX (aHX),
   mHY (aHY),
   mHZ (aHZ)
{
   Normalize();
}


cElHomographie::cElHomographie(const cXmlHomogr & aXmlHom ) :
   mHX (aXmlHom.X()),
   mHY (aXmlHom.Y()),
   mHZ (aXmlHom.Z())
{
   Normalize();
}


cXmlHomogr cElHomographie::ToXml() const
{
   cXmlHomogr aRes;
   aRes.X() = mHX.ToXml();
   aRes.Y() = mHY.ToXml();
   aRes.Z() = mHZ.ToXml();

   return aRes;
}

cElHomographie::cElHomographie(const ElPackHomologue & aPack,bool aL2)
:mHX(cElComposHomographie(1,0,0)),mHY(cElComposHomographie(1,0,0)),mHZ(cElComposHomographie(1,0,0))
{
    ComputeHomFromHomologues(aPack,aL2,mHX,mHY,mHZ);
    Normalize();
}


cElHomographie cElHomographie::Id()
{
   return cElHomographie
          (
                 cElComposHomographie(1,0,0),
                 cElComposHomographie(0,1,0),
                 cElComposHomographie(0,0,1)
          );
}

cElHomographie cElHomographie::Homotie(Pt2dr aP,REAL aSc)
{
   return cElHomographie
          (
                 cElComposHomographie( aSc,    0,   aP.x),
                 cElComposHomographie(   0,  aSc,   aP.y),
                 cElComposHomographie(   0,    0,      1)
          );
}


cElHomographie cElHomographie::RansacInitH(const ElPackHomologue & aPack,int aNbRansac,int aNbMaxPts)
{
   if ((aPack.size()<10)  || (aNbMaxPts<4))
      return cElHomographie(aPack,false);
   

   cRandNParmiQ aRand(aNbMaxPts,aPack.size());
   std::vector<ElCplePtsHomologues> aVCH;

   for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
   {
      if (aRand.GetNext())
      {
          aVCH.push_back(itH->ToCple());
      }
   }

   double anEcMin = 1e30;
   cElHomographie aHMin = cElHomographie::Id();

   while (aNbRansac)
   {
       int aK1 = NRrandom3((int)aVCH.size());
       int aK2 = NRrandom3((int)aVCH.size());
       int aK3 = NRrandom3((int)aVCH.size());
       int aK4 = NRrandom3((int)aVCH.size());
       if ( 
               (aK1!=aK2) && (aK1!=aK3) &&  (aK1!=aK4)
            && (aK2!=aK3) && (aK2!=aK4)
            && (aK3!=aK4)
          )
        {
            aNbRansac--;
            ElPackHomologue aP4;
            aP4.Cple_Add(aVCH[aK1]);
            aP4.Cple_Add(aVCH[aK2]);
            aP4.Cple_Add(aVCH[aK3]);
            aP4.Cple_Add(aVCH[aK4]);

            cElHomographie aSol = cElHomographie(aP4,true);
            double anEcart = 0;
            for (int aKP=0 ; aKP<int(aVCH.size()) ;  aKP++)
            {
                anEcart += euclid(aSol.Direct(aVCH[aKP].P1()),aVCH[aKP].P2());
            }
            anEcart /= aVCH.size();
            if (anEcart<anEcMin)
            {
                anEcMin = anEcart;
                aHMin = aSol;
            }
        }
   }
   std::cout << "ECART H " << anEcMin << "\n";
   return aHMin;
 
}

cElHomographie cElHomographie::FromMatrix(const ElMatrix<REAL> & M)
{
     
   return cElHomographie
          (
                 cElComposHomographie( M(0,0),M(1,0),M(2,0)),
                 cElComposHomographie( M(0,1),M(1,1),M(2,1)),
                 cElComposHomographie( M(0,2),M(1,2),M(2,2))
          );
}

void cElHomographie::ToMatrix(ElMatrix<REAL> &  M) const
{
   M.set_to_size(3,3);
   M(0,0) = mHX.CoeffX(); M(1,0) = mHX.CoeffY(); M(2,0) = mHX.Coeff1();
   M(0,1) = mHY.CoeffX(); M(1,1) = mHY.CoeffY(); M(2,1) = mHY.Coeff1();
   M(0,2) = mHZ.CoeffX(); M(1,2) = mHZ.CoeffY(); M(2,2) = mHZ.Coeff1();
}


cElHomographie cElHomographie::operator * (const cElHomographie & H2) const
{
    ElMatrix<REAL> M1(3,3);
    ElMatrix<REAL> M2(3,3);
    ElMatrix<REAL> Res(3,3);

     this->ToMatrix(M1);
     H2.ToMatrix(M2);

    Res.mul(M1,M2);
    return FromMatrix(Res);
}

void cElHomographie::write(ELISE_fp & aFile) const
{
   mHX.write(aFile);
   mHY.write(aFile);
   mHZ.write(aFile);
}

cElHomographie cElHomographie::read(ELISE_fp & aFile)
{
    cElComposHomographie aHX(0,0,0);
    cElComposHomographie aHY(0,0,0);
    cElComposHomographie aHZ(0,0,0);
    aHX = cElComposHomographie::read(aFile);
    aHY = cElComposHomographie::read(aFile);
    aHZ = cElComposHomographie::read(aFile);

    return cElHomographie(aHX,aHY,aHZ);
}


Pt2dr cElHomographie::Direct(const Pt2dr & aP) const
{
    return Pt2dr(mHX(aP),mHY(aP))/ mHZ(aP);
}


Pt2d<Fonc_Num> cElHomographie::Direct(Pt2d<Fonc_Num> aP) const
{
    Symb_FNum aFZ =  mHZ(aP);
    return Pt2d<Fonc_Num>(mHX(aP)/aFZ,mHY(aP)/aFZ);
}


void cElHomographie::AddForInverse(ElPackHomologue & aPack,Pt2dr aP) const
{
   aPack.Cple_Add(ElCplePtsHomologues(Direct(aP),aP,1.0));
}

cElHomographie cElHomographie::Inverse() const
{
/*
      ElPackHomologue aPack;

      AddForInverse(aPack,Pt2dr( 1, 1));
      AddForInverse(aPack,Pt2dr(-1, 1));
      AddForInverse(aPack,Pt2dr( 1,-1));
      AddForInverse(aPack,Pt2dr(-1,-1));

      return cElHomographie(aPack,true);
*/
      REAL a = mHX.mX, b = mHX.mY, c = mHX.m1;
      REAL d = mHY.mX, e = mHY.mY, f = mHY.m1;
      REAL ALPHA = mHZ.mX, BETA = mHZ.mY;

     cElComposHomographie  IX (    e-BETA * f    ,   -b+ BETA*c     ,   b*f-e*c  );
     cElComposHomographie  IY (    -d+ALPHA*f    ,    a-ALPHA*c     ,   d*c-a*f  );
     cElComposHomographie  IZ (  BETA*d-ALPHA*e  ,  ALPHA*b-BETA*a  ,   a*e-b*d  );

     return cElHomographie(IX,IY,IZ);
}

cElHomographie cElHomographie::MapingChScale(REAL aChSacle) const
{
	return cElHomographie
               (
                   mHX.MulCste(aChSacle),
                   mHY.MulCste(aChSacle),
                   mHZ.MulXY(1.0/aChSacle)
               );
}



cElComposHomographie & cElHomographie::HX() {return mHX;}
cElComposHomographie & cElHomographie::HY() {return mHY;}
cElComposHomographie & cElHomographie::HZ() {return mHZ;}

const cElComposHomographie & cElHomographie::HX() const {return mHX;}
const cElComposHomographie & cElHomographie::HY() const {return mHY;}
const cElComposHomographie & cElHomographie::HZ() const {return mHZ;}

ElMatrix<REAL>  cElHomographie::MatCoordHom() const
{
   ElMatrix<REAL> aRes(3,3);

   mHX.SetCoHom(aRes.data()[0]);
   mHY.SetCoHom(aRes.data()[1]);
   mHZ.SetCoHom(aRes.data()[2]);

   return aRes;
}


double  QuickDist(const Pt2dr & aPt)
{
   double aDx = ElAbs(aPt.x);
   double aDy = ElAbs(aPt.y);
   return   (aDx+aDy + ElMax(aDx,aDy))/ 2.0;
}

void AddPair(ElPackHomologue & aPack,const std::pair<Pt2dr,Pt2dr> &aPair)
{
   aPack.Cple_Add(ElCplePtsHomologues(aPair.first,aPair.second));
}


/*
   "Robust" init of homography :

       1-  Split the homologous point in 4 sector arround cdg (return if one is empty)

       2-  fot aNbTestEstim try :
              - generate a  homography with 4 points, one in each sector
              - memorize the "best" solution as the one minimizing the average of distance (exclufing the
               queue over aPerc)
           Let HR be the result of this Ransac homography

       3- Estimate the homography with an iterative weighted least square estimate,

           - weight  =  1/ (1+4*(E/D)^2)   E=error, D = Dist estimate from 2

      4- Estimate the quality by repetability

*/


/*
template <class TVal> void SplitArrounKthValue(std::vector<TVal> & aV,int aKth)
{
   SplitArrounKthValue(VData(aV),aV.size(),aKth);
}

template <class TVal> TVal MoyKPPVal(std::vector<TVal> & aV,int aKth)
{
   SplitArrounKthValue(aV,aKth);
   return Moy(VData(aV),aKth);
}

template <class TVal> TVal KthVal(std::vector<TVal> & aV,int aKth)
{
    return KthVal(VData(aV),aV.size(),aKth);
}

template <class TVal> TVal MedianeSup(std::vector<TVal> & aV)
{
    return KthVal(aV,aV.size()/2);
}
*/

cElHomographie  cElHomographie::RobustInit(double & aDMIn,double * aQuality,const ElPackHomologue & aPack,bool & Ok ,int aNbTestEstim, double aPerc,int aNbMaxPts)
{
   cElHomographie aRes = cElHomographie::Id();
   Ok = false;
   Pt2dr aCdg(0,0);
   for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
   {
       aCdg = aCdg + itH->P1();
   }

   aCdg = aCdg / double(aPack.size());
   
   std::vector<std::pair<Pt2dr,Pt2dr> > aV00;
   std::vector<std::pair<Pt2dr,Pt2dr> > aV01;
   std::vector<std::pair<Pt2dr,Pt2dr> > aV10;
   std::vector<std::pair<Pt2dr,Pt2dr> > aV11;
   std::vector<std::pair<Pt2dr,Pt2dr> > aVAll;

   int aNbPtsTot = aPack.size();


   int aCpt = 0;
   for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
   {
        Pt2dr aP1 = itH->P1();
        Pt2dr aP2 = itH->P2();
        std::pair<Pt2dr,Pt2dr> aPair(aP1,aP2);

        if (  (((aCpt-1)*aNbMaxPts)/aNbPtsTot)  !=  ((aCpt*aNbMaxPts)/aNbPtsTot))
        {
            aVAll.push_back(aPair);
        }

        if (aP1.x < aCdg.x)
        {
            if (aP1.y < aCdg.y) aV00.push_back(aPair);
            else                aV01.push_back(aPair);
        }
        else
        {
            if (aP1.y < aCdg.y) aV10.push_back(aPair);
            else                aV11.push_back(aPair);
        }
        aCpt++;
   }


   if (aV00.empty()  || aV01.empty()  || aV10.empty()  || aV11.empty()  )
      return aRes;


   aDMIn = 1e30;
   int aNbPts = (int)aVAll.size();
   int aNbKth = ElMax(1,ElMin(aNbPts-1,round_ni((aPerc/100.0) * aNbPts)));
   std::vector<double> aVDist;

   if (aNbMaxPts<aNbPtsTot)
      aNbTestEstim = (aNbTestEstim*aNbPtsTot) / aNbMaxPts;

   // int aKMIN = -1;
   std::vector<double> aVD; // For tuning and show in if(0) ...
   while (aNbTestEstim)
   {
       int aK00 = NRrandom3((int)aV00.size());
       int aK01 = NRrandom3((int)aV01.size());
       int aK10 = NRrandom3((int)aV10.size());
       int aK11 = NRrandom3((int)aV11.size());

       ElPackHomologue aP4;
       AddPair(aP4,aV00[aK00]);
       AddPair(aP4,aV01[aK01]);
       AddPair(aP4,aV10[aK10]);
       AddPair(aP4,aV11[aK11]);

       cElHomographie aSol = cElHomographie(aP4,true);

       aVDist.clear();
       for (int aK=0 ; aK< aNbPts ; aK++)
       {
          Pt2dr aP1 = aVAll[aK].first;
          Pt2dr aP2 = aVAll[aK].second;

/*
          Pt2dr aDif = aP2 -aSol.Direct(aP1);
          double aDx = ElAbs(aDif.x);
          double aDy = ElAbs(aDif.y);
          double aDist =  (aDx+aDy + ElMax(aDx,aDy))/ 2.0;
*/
          double aDist = QuickDist(aP2 -aSol.Direct(aP1));
          aVDist.push_back(aDist);
       }
       
       double aSom = MoyKPPVal(aVDist,aNbKth);
       
       aVD.push_back(aSom);

       //std::cout << "Robust:Hom:SOM = " << aDMIn << " " << aSom << "\n";

       if (aSom <aDMIn)
       {
          aRes = aSol;
          aDMIn = aSom;
       }
       aNbTestEstim--;
   }


   // double aDMinInit = aDMIn;
   ElPackHomologue aPckPds;
   for (int anIterL2 = 0 ; anIterL2 < 4 ; anIterL2++)
   {
       aPckPds = ElPackHomologue();
       aVDist.clear();
       int aCpt = 0;
       for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
       {
           Pt2dr aP1 = itH->P1();
           Pt2dr aP2 = itH->P2();
           double aDist = QuickDist(aP2 -aRes.Direct(aP1));
           aVDist.push_back(aDist);

           double aPds = 1/ (1+ 4*ElSquare(aDist/aDMIn));
           aPckPds.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));
           aCpt++;
       }
       ELISE_ASSERT(aNbPtsTot==aPack.size() ,"KKKKK ????");
       int aKTh = round_ni(aNbPtsTot * (aPerc/100.0));

       ELISE_ASSERT(int(aVDist.size())==aNbPtsTot,"Compat MoyKPPVal/SplitArrounKthValue");
       aDMIn = MoyKPPVal(aVDist,aKTh);

       aRes = cElHomographie(aPckPds,true);
   }

   if (aQuality)
   {
      std::vector<double> aVEstim;
      int aNbTestValid = 71;
      for (int aKTest = 0 ; aKTest <aNbTestValid ; aKTest++)
      {
          ElPackHomologue aPckPdsA;
          ElPackHomologue aPckPdsB;
          cRandNParmiQ  aSelec(aNbPtsTot/2,aNbPtsTot);

          for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
          {
              Pt2dr aP1 = itH->P1();
              Pt2dr aP2 = itH->P2();
              double aDist = QuickDist(aP2 -aRes.Direct(aP1));
              aVDist.push_back(aDist);

              double aPds = 1/ sqrt(1+ ElSquare(aDist/aDMIn));
              // if (NRrandom3() > 0.5) 
              if (aSelec.GetNext())
                  aPckPdsA.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));
              else
                  aPckPdsB.Cple_Add(ElCplePtsHomologues(aP1,aP2,aPds));

           
          }
          cElHomographie aResA = cElHomographie(aPckPdsA,true);
          cElHomographie aResB = cElHomographie(aPckPdsB,true);

          double aSomDist = 0; 
          for (ElPackHomologue::tCstIter itH=aPack.begin() ; itH!=aPack.end() ; itH++)
          {
              Pt2dr aP1 = itH->P1();

              Pt2dr  aQ   = aRes.Direct(aP1);
              Pt2dr  aQA  = aResA.Direct(aP1);
              Pt2dr  aQB  = aResB.Direct(aP1);
              double aDist = (QuickDist(aQ-aQA) + QuickDist(aQ-aQB) + QuickDist(aQB-aQA)) / 3.0;
              aSomDist += aDist;
          }
          aSomDist /= aNbPtsTot;
          aVEstim.push_back(aSomDist);
      }
      *aQuality  = MedianeSup(aVEstim);
   }


   Ok= true;
   return aRes;
}


cElHomographie cElHomographie::SomPondHom(const std::vector<cElHomographie> & aVH,const std::vector<double> & aVP)
{
    ELISE_ASSERT(aVH.size()==aVP.size(),"SomPondHom");
    double aSomHXx = 0;
    double aSomHXy = 0;
    double aSomHX1 = 0;
    double aSomHYx = 0;
    double aSomHYy = 0;
    double aSomHY1 = 0;
    double aSomHZx = 0;
    double aSomHZy = 0;
    double aSomHZ1 = 0;

    double aSomPds = 0;

    for (int aK=0 ; aK<int(aVH.size()) ; aK++)
    {
        double aPds = aVP[aK];

        const cElComposHomographie &  aHX = aVH[aK].HX();
        const cElComposHomographie &  aHY = aVH[aK].HY();
        const cElComposHomographie &  aHZ = aVH[aK].HZ();

        double aMul = aPds / aHZ.Coeff1();


        aSomHXx += aHX.CoeffX() * aMul;
        aSomHXy += aHX.CoeffY() * aMul;
        aSomHX1 += aHX.Coeff1() * aMul;

        aSomHYx += aHY.CoeffX() * aMul;
        aSomHYy += aHY.CoeffY() * aMul;
        aSomHY1 += aHY.Coeff1() * aMul;

        aSomHZx += aHZ.CoeffX() * aMul;
        aSomHZy += aHZ.CoeffY() * aMul;
        aSomHZ1 += aHZ.Coeff1() * aPds;

        aSomPds += aPds;
    }


    return cElHomographie
           (
               cElComposHomographie(aSomHXx/aSomPds,aSomHXy/aSomPds,aSomHX1/aSomPds),
               cElComposHomographie(aSomHYx/aSomPds,aSomHYy/aSomPds,aSomHY1/aSomPds),
               cElComposHomographie(aSomHZx/aSomPds,aSomHZy/aSomPds,aSomHZ1/aSomPds)
           );
}
 


/****************************************************************/
/*                                                              */
/*                     cDistHomographie                         */
/*                                                              */
/****************************************************************/

void  cDistHomographie::Diff(ElMatrix<REAL> &,Pt2dr) const 
{
    ELISE_ASSERT(false,"No cDistHomographie::Diff");
}

Pt2dr cDistHomographie::Direct(Pt2dr aP) const
{
    return mHDir.Direct(aP);
}

bool  cDistHomographie::OwnInverse(Pt2dr & aP) const 
{
   aP = mHInv.Direct(aP);
   return true;
}


cDistHomographie::cDistHomographie(const ElPackHomologue & aPack,bool aL2) :
    mHDir (aPack,aL2),
    mHInv (mHDir.Inverse())
{
}

cDistHomographie::cDistHomographie(const cElHomographie & aHom) :
     mHDir (aHom),
     mHInv (aHom.Inverse())
{
}

cDistHomographie cDistHomographie::MapingChScale(REAL aChSacle) const
{
    return cDistHomographie(mHDir.MapingChScale(aChSacle));
}


ElDistortion22_Gen  * cDistHomographie::D22G_ChScale(REAL aS) const
{
   return new  cDistHomographie(MapingChScale(aS));
}

const cElHomographie & cDistHomographie::Hom() const
{
   return mHDir;
}

/****************************************************************/
/*                                                              */
/*               cDistHomographieRadiale                        */
/*                                                              */
/****************************************************************/

ElDistortion22_Gen  * cDistHomographieRadiale::D22G_ChScale(REAL aS) const
{
   return new cDistHomographieRadiale(MapingChScale(aS));
}

void  cDistHomographieRadiale::Diff(ElMatrix<REAL> &,Pt2dr) const 
{
    ELISE_ASSERT(false,"No cDistHomographie::Diff");
}

cDistHomographieRadiale::cDistHomographieRadiale
(
     const cElHomographie & anHom,
     const ElDistRadiale_PolynImpair & aDRad,
     REAL aRayInv,
     INT  aDeltaDegraInv
)  :
   mHom(anHom),
   mHomInv(anHom.Inverse()),
   mDist(aDRad),
   mDistInv(mDist.DistRadialeInverse(aRayInv,aDeltaDegraInv)),
   mRay (aRayInv),
   mDeg (aDeltaDegraInv)
{
}
cDistHomographieRadiale 
    cDistHomographieRadiale::MapingChScale(REAL aChSacle) const
{
   return cDistHomographieRadiale
          (
              mHom.MapingChScale(aChSacle),
              mDist.MapingChScale(aChSacle),
              mRay * aChSacle,
              mDeg
          );
}

Pt2dr cDistHomographieRadiale::Direct(Pt2dr aP) const  
{
   return mDistInv.Direct(mHom.Direct(mDist.Direct(aP)));
}

bool cDistHomographieRadiale::OwnInverse(Pt2dr & aP) const 
{
   aP = mDistInv.Direct(mHomInv.Direct(mDist.Direct(aP)));
   return true;
}

/*************************************************/






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
