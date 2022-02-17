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



/*************************************************/
/*                                               */
/*               ::                              */
/*                                               */
/*************************************************/

cSystemeCoord SYSCoord(eTypeCoord aTC,int aNb)
{
    cSystemeCoord aRes;
    cBasicSystemeCoord aBSC;
    aBSC.TypeCoord() = aTC;
    for (int aK=0 ; aK<aNb ; aK++)
        aRes.BSC().push_back(aBSC);

    return aRes;
}

void AddTo(cSystemeCoord & aSC1,const cSystemeCoord & aSC2)
{
    for (int aK=0 ; aK<int(aSC2.BSC().size()) ; aK++)
       aSC1.BSC().push_back(aSC2.BSC()[aK]);
}
/*************************************************/
/*                                               */
/*          cSysCoord_GeoC                       */
/*                                               */
/*************************************************/

class cGeoc_SC : public cSysCoord
{
    public :
      Pt3dr ToGeoC(const Pt3dr & aP) const {return aP;}
      Pt3dr FromGeoC(const Pt3dr & aP) const {return aP;}
      Pt3dr OdgEnMetre() const {return Pt3dr(1,1,1);}

      cSystemeCoord ToXML() const
      {
         return SYSCoord(eTC_GeoCentr,1);
      }


      static cGeoc_SC * TheOne()
      {
          static cGeoc_SC * aRes = new cGeoc_SC;
          return aRes;
      }
      void Delete() {}
    private  :
};


/*************************************************/
/*                                               */
/*          cGeoc_WGS4                           */
/*                                               */
/*************************************************/

class cGeoc_WGS4 : public cSysCoord
{
       static const double PtAxe/* = 6356752.3*/;
       static const double GdAxe/* = 6378137.0*/;
    public :
      Pt3dr OdgEnMetre() const {return Pt3dr(PtAxe,PtAxe,1);}
      Pt3dr ToGeoC(const Pt3dr & aP) const
      {
          double longit = ToRadian(aP.x,mUnite);
          double lat = ToRadian(aP.y,mUnite);
          if (mSwap) 
             ElSwap(longit,lat);
          double  h =   aP.z;

          // double ptit_axe=double(6356752.3);
          // double grd_axe=double(6378137.0);
          double ptit_axe= PtAxe;
          double grd_axe= GdAxe;

          double temp = (grd_axe - ptit_axe)/ grd_axe;
          double e2 = 2. * temp - temp*temp;
          double N= grd_axe/(sqrt(1.0-e2*sin(lat)*sin(lat)));

          Pt3dr  aRes
                 (
                     (N+h)*cos(lat)*cos(longit),
                     (N+h)*cos(lat)*sin(longit),
                     (N*(1.-e2)+h)*sin(lat)
                 );

          return aRes;

      }
      Pt3dr FromGeoC(const Pt3dr & aP) const ;
      cSystemeCoord ToXML() const
      {
         return SYSCoord(eTC_WGS84,1);
      }
      static cGeoc_WGS4 * TheOne()
      {
          static cGeoc_WGS4 * aRes = new cGeoc_WGS4;
          return aRes;
      }
      static cGeoc_WGS4 * TheOneDeg()
      {
          static cGeoc_WGS4 * aRes = new cGeoc_WGS4(eUniteAngleDegre);
          return aRes;
      }


      void Delete() {}

      cGeoc_WGS4(const cBasicSystemeCoord & aBSC)
      {
           if (aBSC.AuxRUnite().size()==0)
              mUnite = eUniteAngleRadian;
           else if (aBSC.AuxRUnite().size()==1)
              mUnite = aBSC.AuxRUnite()[0];
           else
           {
               ELISE_ASSERT(false,"cGeoc_WGS4 mre than 1 Unit");
           }
        
           if (aBSC.AuxI().size()==0)
              mSwap = false;
           else if (aBSC.AuxI().size()==1)
              mSwap = (aBSC.AuxI()[0] !=0);
           else
           {
               ELISE_ASSERT(false,"cGeoc_WGS4 mre than 1 AuxI" );
           }
      }

      cGeoc_WGS4( eUniteAngulaire  aUnit = eUniteAngleRadian) :
          mUnite (aUnit),
          mSwap (false)
      {
      }
    private  :
      eUniteAngulaire  mUnite;
      bool             mSwap;
};



/*
        const double cGeoc_WGS4::PtAxe = 6356752.3;
        const double cGeoc_WGS4::GdAxe = 6378137.0;
*/

const double cGeoc_WGS4::PtAxe = 6356752.314140;
const double cGeoc_WGS4::GdAxe = 6378137.0;


       // static const double PtAxe = 6356752.3;
       // static const double GdAxe = 6378137.0;
Pt3dr cGeoc_WGS4::FromGeoC(const Pt3dr & aP) const 
{
   // double b=6356752.3; 
   // double a=6378137.0;
   double b=PtAxe;
   double a=GdAxe;

   double X= aP.x;
   double Y= aP.y;
   double Z= aP.z;
   double temp = (a-b)/a;
   double e2 = 2. * temp - temp*temp;

   double longit=atan(Y/X);
   double lat=atan(Z/(sqrt(X*X+Y*Y)));
   double h = 0;

   int maxiter=500;
   // MPD modif 1e-10 => 1e-15 ; car influence sur la precision des RPC
   double epsilon=1e-15;
   int i=0;
   double delta_lat=1234;
   

   while ( (delta_lat > epsilon)  &&  (i < maxiter) )
   {
      double n= a/(sqrt(1.-e2*sin(lat)*sin(lat)));
      h=sqrt(X*X+Y*Y)/cos(lat)-n;
      double oldlat=lat;
      lat=atan(Z/(sqrt(X*X+Y*Y)*(1-e2*n/(n+h))));
      i=i+1;
      delta_lat=abs(lat-oldlat);
   }

   // std::cout << "NB ITER " << i << " " << delta_lat * 1e20  << "\n";

   if (X < 0)  longit=longit+ PI;

   longit = FromRadian(longit,mUnite);
   lat    = FromRadian(lat   ,mUnite);

   if (mSwap) 
     ElSwap(longit,lat);
   return Pt3dr(longit,lat,h);
}

/*************************************************/
/*                                               */
/*          cCs2Cs                               */
/*                                               */
/*************************************************/
cCs2Cs::cCs2Cs(const std::string  & aStr) :
                     mStr (aStr)
{};


std::vector<Pt3dr> cCs2Cs::Chang(const std::vector<Pt3dr> & aPtsIn) const
{

   std::string aTmpIn = "Proj4Input"+GetUnikId() +".txt";  // Pour exe en //
   FILE * aFPin = FopenNN(aTmpIn,"w","cCs2Cs::Chang");
   for (int aK= 0 ; aK< int(aPtsIn.size()) ; aK++)
   {
       fprintf(aFPin,"%.20f %.20f %.20f\n", aPtsIn.at(aK).x, aPtsIn.at(aK).y, aPtsIn.at(aK).z);
   }
   ElFclose(aFPin);


   std::string aTmpOut = "Proj4Output" + GetUnikId() + ".txt";
   
   std::string aCom = g_externalToolHandler.get("cs2cs").callName() + " " +
                      mStr + " " + aTmpIn + " > " + aTmpOut;

   VoidSystem(aCom.c_str());
   
   ELISE_fp aFOut(aTmpOut.c_str(),ELISE_fp::READ);

   
   std::vector<Pt3dr> aRes;
   char * aLine;
   while ((aLine = aFOut.std_fgets()))
   {
         Pt3dr aP;
         int aNb = sscanf(aLine,"%lf %lf %lf",&aP.x,&aP.y,&aP.z);
         
         ELISE_ASSERT(aNb==3,"Bad Nb value in cCs2Cs::Chang, internal error");
         
         aRes.push_back(aP);
   }
   aFOut.close();

   ELISE_fp::RmFile(aTmpIn);
   ELISE_fp::RmFile(aTmpOut);

   return(aRes);
}

cSystemeCoord cCs2Cs::ToXML() const
{
    cSystemeCoord aRes = SYSCoord(eTC_Proj4,1);

    aRes.BSC()[0].AuxStr().push_back(mStr);
    
    return aRes;
}

/*************************************************/
/*                                               */
/*          cProj4                               */
/*                                               */
/*************************************************/

cProj4::cProj4(const std::string  & aStr,const Pt3dr & aMOdg) :
                     mStr (aStr),
                     mMOdg  (aMOdg)
{};

cProj4 * cProj4::Lambert93()
{
    static cProj4 * aRes =  new cProj4(cProj4::Lambert(46.5,49,44,3,700000,6600000));
    return aRes;
};

Pt3dr cProj4::OdgEnMetre() const 
{
    return mMOdg;
};

cSystemeCoord cProj4::ToXML() const
{
    cSystemeCoord aRes = SYSCoord(eTC_Proj4,1);

    aRes.BSC()[0].AuxR().push_back(mMOdg.x);
    aRes.BSC()[0].AuxR().push_back(mMOdg.y);
    aRes.BSC()[0].AuxR().push_back(mMOdg.z);
    aRes.BSC()[0].AuxStr().push_back(mStr);

    return aRes;
};

void cProj4::Delete()
{};


cProj4  cProj4::Lambert(double aPhi0,double aPhi1,double aPhi2,double aLon0,double aX0,double aY0)
{
   std::string aStr =    std::string(" +proj=lcc ")
                      +  " +lat_1="+ToString(aPhi1)
                      +  " +lat_2="+ToString(aPhi2)
                      +  " +lat_0="+ToString(aPhi0)
                      +  " +lon_0="+ToString(aLon0)
                      +  " +x_0="+ToString(aX0)
                      +  " +y_0="+ToString(aY0)
                      +  " +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 ";

   return cProj4(aStr,Pt3dr(1,1,1));
}


std::vector<Pt3dr> cProj4::ToGeoC(const std::vector<Pt3dr> & aV) const
{
   return Chang(aV,true);
}
std::vector<Pt3dr> cProj4::FromGeoC(const std::vector<Pt3dr> & aV) const
{
   return Chang(aV,false);
}

std::vector<Pt3dr> cProj4::Chang(const std::vector<Pt3dr> & aPtsIn, bool Sens2GeoC) const
{
   cGeoc_WGS4 aWD(eUniteAngleDegre);
   std::string aTmpIn = "Proj4Input"+GetUnikId() +".txt";  // Pour exe en //
   FILE * aFPin = FopenNN(aTmpIn,"w","cProj4::Chang");
   for (int aK= 0 ; aK< int(aPtsIn.size()) ; aK++)
   {
       Pt3dr aP = aPtsIn[aK];
       if (! Sens2GeoC)
       {
           aP =  aWD.FromGeoC(aP);
       }
       // fprintf(aFPin,"%lf %lf %lf\n",aP.x,aP.y,aP.z);
       // fprintf(aFPin,"%Lf %Lf %Lf\n",(long double)aP.x,(long double)aP.y,(long double)aP.z);
       fprintf(aFPin,"%.20f %.20f %.20f\n",aP.x,aP.y,aP.z);
   }
   ElFclose(aFPin);

   std::string aTmpOut = "Proj4Output" + GetUnikId() + ".txt";

   std::string aCom =    std::string( std::string(SYS_CAT) + " " + aTmpIn + " | ") + "\""
						   + g_externalToolHandler.get("proj").callName() + "\"" + (Sens2GeoC?" -I ":" ")
						   + std::string(" -f %.7f ")
						   + mStr
						   + " > " 
						   + aTmpOut;
   
   VoidSystem(aCom.c_str());


   ELISE_fp aFOut(aTmpOut.c_str(),ELISE_fp::READ);

   
   std::vector<Pt3dr> aRes;
   char * aLine;
   while ((aLine = aFOut.std_fgets()))
   {
         Pt3dr aP;
         int aNb = sscanf(aLine,"%lf %lf %lf",&aP.x,&aP.y,&aP.z);
         //std::cout << " ap " << aNb << " " << aP << std::endl;
         ELISE_ASSERT(aNb==3,"Bad Nb value in cProj4::Chang, internal error");
         if (Sens2GeoC)
         {
            aP = aWD.ToGeoC(aP);
         }
         aRes.push_back(aP);
   }
   aFOut.close();

   ELISE_fp::RmFile(aTmpOut);
   ELISE_fp::RmFile(aTmpIn);



   return aRes;
}



/*************************************************/
/*                                               */
/*           RTL                                 */
/*                                               */
/*************************************************/

class cGeoc_RTL : public cSysCoord
{
    public :
      Pt3dr ToGeoC(const Pt3dr & aP) const
      {
           return mRGeocToRTL.ImRecAff(aP);
      }
      Pt3dr FromGeoC(const Pt3dr & aP) const 
      {
           return mRGeocToRTL.ImAff(aP);
      }

      cGeoc_RTL
      (
              const Pt3dr & aP,
              const Pt3dr & aDirX,
              const Pt3dr & aDirY,
              const Pt3dr & aDirZ
      ) :
           mOri        (aP),
           mRGeocToRTL (ElRotation3D(aP,MatFromCol(aDirX,aDirY,aDirZ),true).inv())
      {
      }
      void Delete() {delete this;}
      Pt3dr OdgEnMetre() const {return Pt3dr(1,1,1);}

      cSystemeCoord ToXML() const
      {
         cSystemeCoord aRes = SYSCoord(eTC_RTL,2);

         aRes.BSC()[0].AuxR().push_back(mOri.x);
         aRes.BSC()[0].AuxR().push_back(mOri.y);
         aRes.BSC()[0].AuxR().push_back(mOri.z);

         aRes.BSC()[1].TypeCoord()= eTC_GeoCentr;

         return aRes;
      }

    private :

       Pt3dr        mOri;
       ElRotation3D  mRGeocToRTL;
};

// ElRotation3D RotationCart2RTL(Pt3dr  aP, double aZ)

/*************************************************/
/*                                               */
/*           cSysPolyn                           */
/*                                               */
/*************************************************/

template <class TypeP> void Puhs2V(std::vector<typename TypeP::TypeScal> & aV,const TypeP & aP)
{
    aV.push_back(aP.x);
    aV.push_back(aP.y);
    aV.push_back(aP.z);
}

class cNormCoord
{
     public :
          cNormCoord();
          void Set(const std::vector<Pt3dr> &aVin);
           Pt3dr  ToNorm(const Pt3dr& aP) const
           {
               return Pt3dr ((aP.x-aCDG.x)/aScale.x,(aP.y-aCDG.y)/aScale.y,(aP.z-aCDG.z)/aScale.z);
           }
           Pt3dr  FromNorm(const Pt3dr& aP) const
           {
               return Pt3dr(aP.x*aScale.x+aCDG.x,aP.y*aScale.y+aCDG.y, aP.z*aScale.z+aCDG.z);
           }
           // const Pt3dr & CDG()   const {return aCDG;}
           // const Pt3dr & Scale() const {return aScale;}
           void XmlAdd(cBasicSystemeCoord &) const;
           void InitFromData(const double *& aDVals,int & aNbVals);
     private :
          Pt3dr aCDG;
          Pt3dr aScale;
};

void cNormCoord::InitFromData(const double *& aDVals,int & aNbVals)
{
    ELISE_ASSERT(aNbVals>=6,"cNormCoord::InitFromData No Enough data");
    aCDG   = Pt3dr(aDVals[0],aDVals[1],aDVals[2]);
    aScale = Pt3dr(aDVals[3],aDVals[4],aDVals[5]);

    aDVals +=6;
    aNbVals -= 6;
}

void cNormCoord::XmlAdd(cBasicSystemeCoord & aBSC) const
{
   Puhs2V(aBSC.AuxR(),aCDG);
   Puhs2V(aBSC.AuxR(),aScale);
}

cNormCoord::cNormCoord() :
     aCDG (0,0,0),
     aScale (0,0,0)
{
}

void cNormCoord::Set(const std::vector<Pt3dr> &aVin)
{
    aCDG = Pt3dr(0,0,0);
    aScale = Pt3dr(0,0,0);
    for (int aK=0 ;aK<int(aVin.size()) ; aK++)
    {
        Pt3dr aP = aVin[aK];
        aCDG = aCDG +  aP;
    }
    aCDG = aCDG / (double)(aVin.size());


    for (int aK=0 ;aK<int(aVin.size()) ; aK++)
    {
        Pt3dr aP = aVin[aK];
        aScale = aScale + (aP-aCDG).AbsP();
    }
    aScale = aScale /(double)(aVin.size());

/*
std::cout << "TRICHE " << aCDG << " " << aScale << "\n";
getchar();
     aCDG = Pt3dr(0,0,0);
     aScale = Pt3dr(1,1,1);
*/

}


class cOneCoorSysPolyn
{
     public :
         cOneCoorSysPolyn(Pt3di aDeg) ;
         void SetPtIn(const Pt3dr& aPtIn);
         void  AddObs(const Pt3dr& aPtIn,double aVal);

         void Solve();

         double GetVal(const Pt3dr &);
         // const Pt3di & Deg() const {return mDeg;}
         // double * Data()  const {return mDS;}
         void XmlAdd(cBasicSystemeCoord &,bool WithDegr) const;
         void InitFromData(const double *& aDVals,int & aNbVals);

     private :
          std::vector<double> mPX;
          std::vector<double> mPY;
          std::vector<double> mPZ;
          int                 mNbDeg;
          Pt3di               mDeg;
          std::vector<double> mCoord;
          L2SysSurResol *     mSys;
          Im1D_REAL8          mSol;
          double *            mDS;
};

void cOneCoorSysPolyn::InitFromData(const double *& aDVals,int & aNbVals)
{
   ELISE_ASSERT(aNbVals>=mNbDeg,"cOneCoorSysPolyn::InitFromData Not Enouh Vals");
   mSol = Im1D_REAL8(mNbDeg);
   mDS = mSol.data();
   for (int aK=0 ; aK<mNbDeg ; aK++)
       mDS[aK] = aDVals[aK];

   aDVals+= mNbDeg;
   aNbVals -= mNbDeg;
}

void cOneCoorSysPolyn::XmlAdd(cBasicSystemeCoord & aBSC,bool WithDegr) const
{
   if (WithDegr) 
   {
      Puhs2V(aBSC.AuxI(),mDeg);
   }
   for (int aK=0 ; aK<mNbDeg ; aK++)
   {
      aBSC.AuxR().push_back(mDS[aK]);
   }
}

void cOneCoorSysPolyn::Solve()
{
    ELISE_ASSERT(mSys,"cOneCoorSysPolyn::Solve");
    mSol = mSys->Solve(0);
    mDS = mSol.data();
}

cOneCoorSysPolyn::cOneCoorSysPolyn
(
     Pt3di aDeg
)  :
   mPX     (1+aDeg.x),
   mPY     (1+aDeg.y),
   mPZ     (1+aDeg.z),
   mNbDeg  ((int)(mPX.size() * mPY.size() * mPZ.size())),
   mDeg    (aDeg),
   mCoord  (mNbDeg),
   mSys    (0),
   mSol    (1),
   mDS     (0)
{
}

void cOneCoorSysPolyn::AddObs(const Pt3dr& aPtIn,double aVal)
{
    if (mSys==0)
    {
       mSys = new L2SysSurResol(mNbDeg,false);
    }
    SetPtIn(aPtIn);
    mSys->AddEquation(1.0,&mCoord[0],aVal);
}

void InitPow(std::vector<double> & aVect,double aVal)
{
    aVect[0] = 1.0;
    for (int aK=1 ; aK<int(aVect.size()) ; aK++)
        aVect[aK] = aVal * aVect[aK-1];
}

void cOneCoorSysPolyn::SetPtIn(const Pt3dr& aPtIn)
{
    InitPow(mPX,aPtIn.x);
    InitPow(mPY,aPtIn.y);
    InitPow(mPZ,aPtIn.z);

    int aCpt=0;
    for (int aKx=0 ; aKx<=mDeg.x ; aKx++)
    {
        for (int aKy=0 ; aKy<=mDeg.y ; aKy++)
        {
            for (int aKz=0 ; aKz<=mDeg.z ; aKz++)
            {
                 mCoord[aCpt] = mPX[aKx] * mPY[aKy] * mPZ[aKz];
                 aCpt++;
            }
        }
    }
}

double cOneCoorSysPolyn::GetVal(const Pt3dr & aP)
{
    SetPtIn(aP);
    double aRes=0;
    for (int aK=0 ; aK<int(mCoord.size()) ; aK++)
       aRes += mDS[aK] * mCoord[aK];
   return aRes;
}


class cOneSenSysPolyn
{
     public :
            cOneSenSysPolyn(Pt3di aDegX,Pt3di aDegY,Pt3di aDegZ);
            void SetByApr(const std::vector<Pt3dr> &,const std::vector<Pt3dr> &);
            Pt3dr  GetVal(const  Pt3dr & aP) const;
/*
            const cOneCoorSysPolyn & CX()   const {return mCX;}
            const cOneCoorSysPolyn & CY()   const {return mCY;}
            const cOneCoorSysPolyn & CZ()   const {return mCZ;}
            const cNormCoord &       CIn()  const {return mCIn;}
            const cNormCoord &       COut() const {return mCOut;}
*/

            void XmlAdd(cBasicSystemeCoord &,bool WithDeg) const;
            void InitFromData(const double *& aDVals,int & aNbVals);
     private :
          cNormCoord  mCIn;
          cNormCoord  mCOut;
          mutable cOneCoorSysPolyn  mCX;
          mutable cOneCoorSysPolyn  mCY;
          mutable cOneCoorSysPolyn  mCZ;
};


void cOneSenSysPolyn::XmlAdd(cBasicSystemeCoord & aSC,bool WithDeg) const
{
    mCIn.XmlAdd(aSC);
    mCOut.XmlAdd(aSC);
    mCX.XmlAdd(aSC,WithDeg);
    mCY.XmlAdd(aSC,WithDeg);
    mCZ.XmlAdd(aSC,WithDeg);
}

void cOneSenSysPolyn::InitFromData(const double *& aDVals,int &aNbVals)
{
    mCIn.InitFromData(aDVals,aNbVals);
    mCOut.InitFromData(aDVals,aNbVals);
    mCX.InitFromData(aDVals,aNbVals);
    mCY.InitFromData(aDVals,aNbVals);
    mCZ.InitFromData(aDVals,aNbVals);
}

Pt3dr  cOneSenSysPolyn::GetVal(const  Pt3dr & aP) const
{
    Pt3dr aPtIn  =  mCIn.ToNorm(aP);

    return mCOut.FromNorm(Pt3dr(mCX.GetVal(aPtIn),mCY.GetVal(aPtIn),mCZ.GetVal(aPtIn)));
}


cOneSenSysPolyn::cOneSenSysPolyn
(
     Pt3di aDegX,
     Pt3di aDegY,
     Pt3di aDegZ
)  :
   mCX    (aDegX),
   mCY    (aDegY),
   mCZ    (aDegZ)
{
}


void cOneSenSysPolyn::SetByApr
     (
           const std::vector<Pt3dr> & aVin,
           const std::vector<Pt3dr> & aVout
     )
{
    ELISE_ASSERT(aVin.size()==aVout.size(),"Taille Diff in cOneSenSysPolyn::cOneSenSysPolyn");
    mCIn.Set(aVin);
    mCOut.Set(aVout);

    for (int aK=0 ; aK<int(aVin.size()) ; aK++)
    {
//std::cout << "KKKKKKKKKKK  " <<aK << aVin[aK] << " " << aVout[aK] <<  <<  "\n";
// std::cout << mCIn.FromNorm(mCIn.ToNorm(aVin[aK]))  << " " << aVin[aK] << "\n";   :::  OK
        Pt3dr aPtIn  =  mCIn.ToNorm(aVin[aK]);
        Pt3dr aPtOut = mCOut.ToNorm(aVout[aK]);
        mCX.AddObs(aPtIn,aPtOut.x);
        mCY.AddObs(aPtIn,aPtOut.y);
        mCZ.AddObs(aPtIn,aPtOut.z);
    }
    mCX.Solve();
    mCY.Solve();
    mCZ.Solve();
}






// A partir d'un systeme A (eventuellement Ident)  et d'exemples transforamnt A en B,
// calcule une approcimxtiona polynoimale de B;

class  cSysCoordPolyn : public cSysCoord
{
   public :
      void InitFromData(const double *& aDVals,int & aNbVals);
      Pt3dr ToGeoC(const Pt3dr & aP) const 
      {
         return mSysIn->ToGeoC(mSysPolInv.GetVal(aP));
      }
      Pt3dr FromGeoC(const Pt3dr & aP) const 
      {
         return mSysPolDir.GetVal(mSysIn->FromGeoC(aP));
      }
      Pt3dr OdgEnMetre() const {return Pt3dr(1,1,1);}

      cSysCoordPolyn
      (
           Pt3di aDegX,
           Pt3di aDegY,
           Pt3di aDegZ,
           cSysCoord * aSysIn
      );

      
      void InitByApr(
                const std::vector<Pt3dr> & aVin,
                const std::vector<Pt3dr> & aVout
           );
      cSystemeCoord ToXML() const;
      void Delete() {}
      // cOneSenSysPolyn & Dir() {return mSysPolDir;}
      // cSysCoord *      SysIn() {return mSysIn;}
   private :
      cSysCoord *      mSysIn;   // B->GeoC
      cOneSenSysPolyn  mSysPolDir; // A ->B
      cOneSenSysPolyn  mSysPolInv;  // B->A
};



cSystemeCoord cSysCoordPolyn::ToXML() const
{
   cSystemeCoord aRes = SYSCoord(eTC_Polyn,1);
   cSystemeCoord aSIn = mSysIn->ToXML();
   AddTo(aRes,aSIn);
   mSysPolDir.XmlAdd(aRes.BSC()[0],true);
   mSysPolInv.XmlAdd(aRes.BSC()[0],false);
   // ELISE_ASSERT(false,"cSysCoordPolyn::ToXML");
   return aRes;
}

void cSysCoordPolyn::InitFromData(const double *& aDVals,int & aNbVals)
{
   mSysPolDir.InitFromData(aDVals,aNbVals);
   mSysPolInv.InitFromData(aDVals,aNbVals);
}

cSysCoordPolyn::cSysCoordPolyn
(
     Pt3di aDegX,
     Pt3di aDegY,
     Pt3di aDegZ,
     cSysCoord * aSysIn
) :
   mSysIn  (aSysIn),
   mSysPolDir (aDegX,aDegY,aDegZ),
   mSysPolInv (aDegX,aDegY,aDegZ)
{
}


void cSysCoordPolyn::InitByApr
     (
                const std::vector<Pt3dr> & aVin,
                const std::vector<Pt3dr> & aVout
     )
{
   std::vector<Pt3dr> aVStabIn;
   for (int aK=0 ; aK<int(aVin.size()) ; aK++)
   {
       aVStabIn.push_back(mSysIn->FromGeoC(mSysIn->ToGeoC(aVin[aK])));
   }
   mSysPolDir.SetByApr(aVStabIn,aVout);
   mSysPolInv.SetByApr(aVout,aVStabIn);
}


/*************************************************/
/*                                               */
/*           cSysCoord                           */
/*                                               */
/*************************************************/

cSysCoord * cSysCoord::GeoC() { return cGeoc_SC::TheOne(); }
cSysCoord * cSysCoord::WGS84() { return cGeoc_WGS4::TheOne(); }
cSysCoord * cSysCoord::WGS84Degre() { return cGeoc_WGS4::TheOneDeg(); }

cSysCoordPolyn * cSysCoord::TypedModelePolyNomial 
            (
                  Pt3di aDegX,
                  Pt3di aDegY,
                  Pt3di aDegZ,
                  cSysCoord * aSysIn,  
                  const std::vector<Pt3dr> & aVin,
                  const std::vector<Pt3dr> & aVout
            )
{
   cSysCoordPolyn * aRes = new cSysCoordPolyn(aDegX,aDegY,aDegZ,aSysIn);
   aRes->InitByApr(aVin,aVout);
   return aRes;
}

cSysCoord * cSysCoord::ModelePolyNomial 
            (
                  Pt3di aDegX,
                  Pt3di aDegY,
                  Pt3di aDegZ,
                  cSysCoord * aSysIn,  
                  const std::vector<Pt3dr> & aVin,
                  const std::vector<Pt3dr> & aVout
            )
{
   return TypedModelePolyNomial(aDegX,aDegY,aDegZ,aSysIn,aVin,aVout);
}

std::vector<double>  VecCorrecUnites(const std::vector<double> & aV,const std::vector<eUniteAngulaire> &aVU)
{
   std::vector<double> aRes;
   for (int aK=0 ; aK<int(aV.size())  ; aK++)
   {
       if (aK>= int(aVU.size()))
       {
           aRes.push_back(aV[aK]);
       }
       else
       {
           aRes.push_back(ToRadian(aV[aK],aVU[aK]));
       }
   }
   return aRes;
}

cSysCoord * cSysCoord::RTL(Pt3d<double> const& aGOri)  // Origine Geocentrique
{
   cSysCoord * aSW = cSysCoord::WGS84();

   Pt3dr aWOri  = aSW->FromGeoC(aGOri);  // Origine WGS4 pour calculer le plan tanget

   Pt3dr  aU = WGS84()->OdgEnMetre();
   Pt3dr  aDirX = vunit(aSW->ToGeoC(aWOri+Pt3dr(1/aU.x,0,0))-aGOri);
   Pt3dr  aDirY = vunit(aSW->ToGeoC(aWOri+Pt3dr(0,1/aU.y,0))-aGOri);
   Pt3dr  aDirZ = vunit(aSW->ToGeoC(aWOri+Pt3dr(0,0,1/aU.z))-aGOri);

   aDirY = vunit(aDirZ ^aDirX);
   aDirX = vunit(aDirY ^aDirZ);

    cGeoc_RTL * aRes = new cGeoc_RTL(aGOri,aDirX,aDirY,aDirZ);
    return aRes;
}

cSysCoord * cSysCoord::FromXML
                        (
                                   const cBasicSystemeCoord * & aVBSC,
                                   int & aNbB,
                                   const char * aDir
                       )
{
   ELISE_ASSERT(aNbB>=1,"cSysCoord::FromXML no cBasicSystemeCoord");
   if (aVBSC[0].ByFile().Val())
   {
        ELISE_ASSERT
        (
            aVBSC[0].AuxStr().size() >=1,
            "No String with File BasicSystemeCoord"
        );
        eTypeCoord aTC0 = aVBSC[0].TypeCoord();
        std::string aNF = aVBSC[0].AuxStr()[0];
        if (aDir) 
        {
           aNF = std::string(aDir) + aNF;
        }
        std::string aNameTag = (aVBSC[0].AuxStr().size() >=2) ? aVBSC[0].AuxStr()[1] : "SystemeCoord";
        cSystemeCoord aXmlSC= StdGetObjFromFile<cSystemeCoord>
                              (
                                  aNF,
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  aNameTag,
                                  "SystemeCoord"
                              );
         
         ELISE_ASSERT
         (
             (aTC0==eTC_Unknown) || (aTC0==aXmlSC.BSC()[0].TypeCoord()),
             "Incoherence in cSysCoord::TypeCoord()"
         );
         cSysCoord *  aRes = cSysCoord::FromXML(aXmlSC,aDir);
         aVBSC++; aNbB--;
         return aRes;
   }
   if (aVBSC[0].TypeCoord() == eTC_WGS84)
   {
      cSysCoord * aRes = new cGeoc_WGS4(aVBSC[0]);
      aVBSC++; aNbB--;
      return aRes;
   }

   if (aVBSC[0].TypeCoord() == eTC_GeoCentr)
   {
      aVBSC++; aNbB--;
      return GeoC();
   }

   if (aVBSC[0].TypeCoord() == eTC_Lambert93)
   {
      aVBSC++; aNbB--;
      return  cProj4::Lambert93();
   // cProj4  cProj4::Lambert(double aPhi0,double aPhi1,double aPhi2,double aLon0,double aX0,double aY0)
   }

   if (aVBSC[0].TypeCoord() == eTC_LambertCC)
   {
      ELISE_ASSERT(aVBSC[0].AuxI().size() ==1,"Bad AuxI Size in eTC_LambertCC");
      int aPhi = aVBSC[0].AuxI()[0];
      int aK = aPhi-42;
      aVBSC++; aNbB--;
      return new cProj4(cProj4::Lambert(aPhi,aPhi+0.75,aPhi-0.75,3,1.7e6,1.2e6+aK*1e6));
   }


   if (aVBSC[0].TypeCoord() == eTC_Proj4)
   {
      std::string aCom;
      for (int aK=0 ; aK<int(aVBSC[0].AuxStr().size()); aK++)
      {
          aCom = aCom +  " "  +  aVBSC[0].AuxStr()[aK];
      }

      double aVOdg[3] = {1,1,1};
      for (int aK=0 ; aK<ElMin(3,int(aVBSC[0].AuxR().size())); aK++)
      {
          aVOdg[aK] = aVBSC[0].AuxR()[aK];
      }
      Pt3dr anOdg(aVOdg[0],aVOdg[1],aVOdg[2]);

      aVBSC++; aNbB--;
      return new cProj4(aCom,anOdg);
   }


   if (aVBSC[0].TypeCoord()== eTC_RTL)
   {
        
        const std::vector<double> & aVBas = aVBSC[0].AuxR();
        ELISE_ASSERT(aVBas.size()==3,"Bad size in SysOriRTL::AuxR");
        std::vector<double> aV = VecCorrecUnites(aVBas,aVBSC[0].AuxRUnite());
        double aX =  aV[0];
        double aY =  aV[1];
        double aZ =  aV[2];

        aVBSC++; aNbB--;
        Pt3dr aGOri;
        {
            cSysCoord * aSRtl = cSysCoord::FromXML(aVBSC,aNbB,aDir);
            aGOri =  aSRtl->ToGeoC(Pt3dr(aX,aY,aZ));
            aSRtl->Delete();
        }
        cSysCoord * aRes =  cGeoc_RTL::RTL(aGOri);  // Origine Geocentrique
/*
        cSysCoord * aSW = cSysCoord::WGS84();

        Pt3dr aWOri  = aSW->FromGeoC(aGOri);

        Pt3dr  aU = WGS84()->OdgEnMetre();
        Pt3dr  aDirX = vunit(aSW->ToGeoC(aWOri+Pt3dr(1/aU.x,0,0))-aGOri);
        Pt3dr  aDirY = vunit(aSW->ToGeoC(aWOri+Pt3dr(0,1/aU.y,0))-aGOri);
        Pt3dr  aDirZ = vunit(aSW->ToGeoC(aWOri+Pt3dr(0,0,1/aU.z))-aGOri);

        aDirY = vunit(aDirZ ^aDirX);
        aDirX = vunit(aDirY ^aDirZ);

        cGeoc_RTL * aRes = new cGeoc_RTL(aGOri,aDirX,aDirY,aDirZ);
*/
/*
   NE PAS EFFACER AVANT 2013 !! Version ancienne , tangente au system de codage de l'origine
   remplacer par tgt au WGS84
        Pt3dr  aU = aSRtl->OdgEnMetre();
        Pt3dr anOri =  aSRtl->ToGeoC(Pt3dr(aX,aY,aZ));

        Pt3dr  aDirX =  vunit(aSRtl->ToGeoC(Pt3dr(aX+1/aU.x,aY,aZ))- anOri);
        Pt3dr  aDirY =  vunit(aSRtl->ToGeoC(Pt3dr(aX,aY+1/aU.y,aZ))- anOri);
        Pt3dr  aDirZ =  vunit(aSRtl->ToGeoC(Pt3dr(aX,aY,aZ+1/aU.z))- anOri);

        // std::cout << "SCAL " << scal(aDirX,aDirY) <<  " " << scal(aDirX,aDirZ) <<  " " << scal(aDirZ,aDirY) <<  "\n";
        // Avec WGS, les vecteurs sont deja Orthog, mais bon ... 
        // Au cas, ou on conserve le plan "horiz" et le "nord" de ce plan
         aDirZ = vunit(aDirX ^ aDirY);
         aDirX = vunit(aDirY ^aDirZ);
*/

        // double aEpsInit = 1e-5; 





        return aRes;
   }

   if (aVBSC[0].TypeCoord()== eTC_Polyn)
   {

      const int    * aDI = &(aVBSC[0].AuxI()[0]);
      int aNbI = (int)aVBSC[0].AuxI().size();
      ELISE_ASSERT(aNbI==9,"Bad int size in  cSysCoord::FromXML  eTC_Polyn");
      Pt3di aNbX(aDI[0],aDI[1],aDI[2]);
      Pt3di aNbY(aDI[3],aDI[4],aDI[5]);
      Pt3di aNbZ(aDI[6],aDI[7],aDI[8]);

      const double * aDR = &(aVBSC[0].AuxR()[0]);
      int aNbR = (int)aVBSC[0].AuxR().size();

      aVBSC++; aNbB--;
      cSysCoord * aSysIn = cSysCoord::FromXML(aVBSC,aNbB,aDir);

      cSysCoordPolyn * aRes = new cSysCoordPolyn(aNbX,aNbY,aNbZ,aSysIn);
      
      aRes->InitFromData(aDR,aNbR);

       ELISE_ASSERT(aNbR==0,"Chek Sum Pb in cSysCoord::FromXML eTC_Polyn");

      return aRes;
   }
   

    std::cout << "For sys=" << eToString(aVBSC[0].TypeCoord()) << "\n";
    ELISE_ASSERT(false,"cSysCoord::FromXML");
    return 0;
}

cSysCoord * cSysCoord::FromXML(const cSystemeCoord & aSC,const char * aDir)
{
   
   const cBasicSystemeCoord * aVBSC = &(aSC.BSC()[0]);
   int aNbB = (int)aSC.BSC().size();
 
// std::cout << "NB IN " << aNbB<< "\n";
   cSysCoord * aRes = cSysCoord::FromXML(aVBSC,aNbB,aDir);
// std::cout << "NB OUT " << aNbB<< "\n";
   ELISE_ASSERT(aNbB==0,"cSysCoord::FromXML Nb Sys Error");
   return aRes;
}

#define TheNbSysPredef 4
const std::string NameSysPredef[TheNbSysPredef] =
                  {
                       "GeoC",
                       "WGS84",
                       "DegreeWGS84",
                       "Lambert93"
                  };

cSysCoord * cSysCoord::FromFile(const std::string & aNF,const std::string & aNameTag)
{
   std::string aNBasic = NameWithoutDir(aNF);
   if (aNBasic==NameSysPredef[0])        return GeoC();
   if (aNBasic==NameSysPredef[1])       return WGS84();
   if (aNBasic==NameSysPredef[2]) return cGeoc_WGS4::TheOneDeg();
   if (aNBasic==NameSysPredef[3])   return cProj4::Lambert93();

   if (!ELISE_fp::exist_file(aNF))
   {
      std::cout << "Bad Sys coord  for " << aNF << "\n";
      std::cout << " not an existing file, not an allowed value \n";
      for (int aK=0 ; aK<TheNbSysPredef ; aK++)
          std::cout << "   " << NameSysPredef[aK] << "\n";
      
      ELISE_ASSERT(false,"cSysCoord::FromFile");
   }


   cSystemeCoord  aCS = StdGetObjFromFile<cSystemeCoord>
                              (
                                  aNF,
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  aNameTag,
                                  "SystemeCoord"
                              );

    std::string aDir = DirOfFile(aNF);
    return FromXML(aCS,aDir.c_str());
}


Pt3dr cSysCoord::FromSys2This(const cSysCoord & aSys,const Pt3dr & aP) const
{
  return FromGeoC(aSys.ToGeoC(aP));
}

ElMatrix<double> cSysCoord::JacobSys2This(const cSysCoord & aS2,const Pt3dr & aP,const Pt3dr & E) const
{
     return JacobFromGeoc(aS2.ToGeoC(aP),E) * aS2.JacobToGeoc(aP,E);
}


cSysCoord::~cSysCoord()
{
}

Pt3dr  cSysCoord::Transfo(const Pt3dr & aP, bool SensToGeoC) const
{
   return SensToGeoC  ? ToGeoC(aP) : FromGeoC(aP);
}

std::vector<Pt3dr>  cSysCoord::Transfo(const std::vector<Pt3dr> & aV, bool SensToGeoC) const
{
   return SensToGeoC  ? ToGeoC(aV) : FromGeoC(aV);
}



std::vector<ElMatrix<double> > cSysCoord::Jacobien
                               (
                                    const  std::vector<Pt3dr> & aV0,
                                    const Pt3dr& E,
                                    bool SensToGeoC,
                                    std::vector<Pt3dr> * aResPts
                               ) const
{
     if (aResPts) 
       aResPts->clear();
     std::vector<ElMatrix<double> > aResMatr;
     Pt3dr aDx(E.x,0,0);
     Pt3dr aDy(0,E.y,0);
     Pt3dr aDz(0,0,E.z);

     std::vector<Pt3dr> aV;

     for (int aK=0 ; aK<int(aV0.size()) ; aK++)
     {
         Pt3dr aP = aV0[aK];
         aV.push_back(aP+aDx);
         aV.push_back(aP-aDx);
         aV.push_back(aP+aDy);
         aV.push_back(aP-aDy);
         aV.push_back(aP+aDz);
         aV.push_back(aP-aDz);
         if (aResPts)
            aV.push_back(aP);
     }

     aV = Transfo(aV,SensToGeoC);
   
     for (int aK=0 ; aK<int(aV0.size()) ; aK++)
     {
         int aK6_7 = aK * (6 + (aResPts!=0));
         Pt3dr aDerivX = (aV[aK6_7+1]-aV[aK6_7+0]) / (2*E.x);
         Pt3dr aDerivY = (aV[aK6_7+3]-aV[aK6_7+2]) / (2*E.y);
         Pt3dr aDerivZ = (aV[aK6_7+5]-aV[aK6_7+4]) / (2*E.z);


         aResMatr.push_back(MatFromCol(aDerivX,aDerivY,aDerivZ));
         if (aResPts)
         {
            aResPts->push_back(aV[aK6_7+6]);
         }
     }

    return aResMatr;
}




ElMatrix<double>  cSysCoord::Jacobien(const  Pt3dr & aP,const Pt3dr& E,bool SensToGeoC) const
{
    std::vector<Pt3dr> aV0;
    aV0.push_back(aP);
   
    std::vector<ElMatrix<double> > aVRes = Jacobien(aV0,E,SensToGeoC);

    return aVRes[0];
}



ElMatrix<double> cSysCoord::JacobFromGeoc(const Pt3dr & aP,const Pt3dr& Epsilon) const
{
   return Jacobien(aP,Epsilon,false);
}

ElMatrix<double> cSysCoord::JacobToGeoc(const Pt3dr & aP,const Pt3dr& Epsilon) const
{
   return Jacobien(aP,Epsilon,true);
}






void TestPolyn
     ( 
           Pt3di aDegX,
           Pt3di aDegY,
           Pt3di aDegZ,
           const std::vector<Pt3dr> & aVAprIn,
           const std::vector<Pt3dr> & aVAprOut,
           const std::vector<Pt3dr> & aVTestIn,
           const std::vector<Pt3dr> & aVTestOut
     )
{
    cOneSenSysPolyn aOSP(aDegX,aDegY,aDegZ);
    aOSP.SetByApr(aVAprIn,aVAprOut);
    
    double aDMoy=0;
    Pt3dr  aPMax(0,0,0);
    for (int aK=0 ; aK<int (aVTestIn.size()) ; aK++)
    {
        Pt3dr aDif = aVTestOut[aK] - aOSP.GetVal(aVTestIn[aK]);
        double aDist = euclid(aDif);
        aPMax = Sup(aPMax,aDif.AbsP());
        
        aDMoy += aDist;
    }
    std::cout << "Max = " << aPMax  << " Moy = " << aDMoy/aVTestIn.size() << "\n";


   cSysCoord *  aW = cSysCoord::WGS84();
   cSysCoordPolyn *  aSC = cSysCoord::TypedModelePolyNomial
                      (
                                    Pt3di(4,4,0),
                                    Pt3di(4,4,0),
                                    Pt3di(0,0,1),
                                    aW,
                                    aVAprIn,aVAprOut
                      );

  Pt3dr  aPM1(0,0,0);
  Pt3dr  aPM2(0,0,0);
  for (int aK=0 ; aK<int(aVTestIn.size()) ; aK++)
   {
      Pt3dr aG = aW->ToGeoC(aVTestIn[aK]);
      Pt3dr aUTM = aSC->FromGeoC(aG);

      aPM1 = Sup(aPM1,(aUTM-aVTestOut[aK]).AbsP());
      aPM2 = Sup(aPM2,(aG-aSC->ToGeoC(aUTM)).AbsP());
   }
   std::cout << aPM1 << aPM2 << "\n";

}
#if (0)
#endif


Pt3dr cSysCoord::ToGeoC(const Pt3dr & aP) const
{
    std::vector<Pt3dr> aV;
    aV.push_back(aP);
    aV = ToGeoC(aV);
    return aV[0];
}
std::vector<Pt3dr> cSysCoord::ToGeoC(const std::vector<Pt3dr> & aV) const 
{
   std::vector<Pt3dr> aRes;
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
      aRes.push_back(ToGeoC(aV[aK]));
   return aRes;
}


Pt3dr cSysCoord::FromGeoC(const Pt3dr & aP) const
{
    std::vector<Pt3dr> aV;
    aV.push_back(aP);
    aV = FromGeoC(aV);
    return aV[0];
}
std::vector<Pt3dr> cSysCoord::FromGeoC(const std::vector<Pt3dr> & aV) const 
{
   std::vector<Pt3dr> aRes;
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
      aRes.push_back(FromGeoC(aV[aK]));
   return aRes;
}



/**************************************************/
/*                                                */
/*          cGeoRefRasterFile                     */
/*                                                */
/**************************************************/


cGeoRefRasterFile::cGeoRefRasterFile
(
    const NS_ParamChantierPhotogram::cXmlGeoRefFile & aXG,
    const char * aDir
)  :
   mSys (   aXG.SysCo().IsInit() ?
            cSysCoord::FromXML(aXG.SysCo().Val(),aDir) :
            0
        ),
   mHasZMoy  (aXG.ZMoyen().IsInit()),
   mOriXY    (aXG.RefPlani().Origine()),
   mResolXY  (aXG.RefPlani().Resolution()),
   mOriZ     (   mHasZMoy   ?
                 aXG.GestionAltimetrie().ZMoyen().Val() :
                 aXG.GestionAltimetrie().RefAlti().Val().Origine() 
             ),
   mResolZ   (   mHasZMoy   ?
                 0.0        :
                 aXG.GestionAltimetrie().RefAlti().Val().Resolution() 
             )
{
}

void  cGeoRefRasterFile::AssertZMoy() const
{
   ELISE_ASSERT(mHasZMoy,"No Zmoyen in XmlGeoRefFile, cGeoRefRasterFile::To3D");
}


Pt3dr cGeoRefRasterFile::Raster2DTo3D(const Pt2dr & aP) const
{
   AssertZMoy();
   return Pt3dr(aP.x,aP.y,0.0);
}

double cGeoRefRasterFile::ZMoyen() const
{
   AssertZMoy();
   return mOriZ;
}


void cGeoRefRasterFile::AssertSysCo() const
{
   ELISE_ASSERT(mSys,"No Sys Co in XmlGeoRefFile, cGeoRefRasterFile::AssertSysCo");
}

Pt3dr cGeoRefRasterFile::File2Loc(const Pt3dr & aPRas) const
{
  
   return Pt3dr
          (
               mOriXY.x + aPRas.x * mResolXY.x,
               mOriXY.y + aPRas.y * mResolXY.y,
               mOriZ    + aPRas.z * mResolZ
          );
}
Pt3dr cGeoRefRasterFile::File2Loc(const Pt2dr & aPRas) const
{
    return File2Loc(Raster2DTo3D(aPRas));
}
Pt3dr cGeoRefRasterFile::Loc2File(const Pt3dr & aP) const
{
   double aX = (aP.x-mOriXY.x) / mResolXY.x;
   double aY = (aP.y-mOriXY.y) / mResolXY.y;
   double aZ = mResolZ ? ((aP.z-mOriZ)/mResolZ) : 0.0;
   return Pt3dr(aX,aY,aZ);

}
Pt3dr cGeoRefRasterFile::Geoc2File(const Pt3dr & aP) const
{
   AssertSysCo();
   return Loc2File(mSys->FromGeoC(aP));
}


Pt3dr cGeoRefRasterFile::File2GeoC(const Pt3dr & aP) const
{
    AssertSysCo();
    return  mSys->ToGeoC(File2Loc(aP));
}

Pt3dr cGeoRefRasterFile::File2GeoC(const Pt2dr & aP) const
{
   return File2GeoC(Raster2DTo3D(aP));
}
cGeoRefRasterFile * cGeoRefRasterFile::FromFile(const std::string & aNF,const std::string & aNameTag)
{
   cXmlGeoRefFile  aXGRF = StdGetObjFromFile<cXmlGeoRefFile>
                              (
                                  aNF,
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  aNameTag,
                                  "XmlGeoRefFile"
                              );

    std::string aDir = DirOfFile(aNF);
    return new cGeoRefRasterFile(aXGRF,aDir.c_str());
}


/***********************************************/
/*                                             */
/*             cChSysCo                        */
/*                                             */
/***********************************************/

/*
cChSysCo::cChSysCo(const cChangementCoordonnees & aCC,const std::string & aDir)  :
   mSrc  (cSysCoord::FromXML(aCC.SystemeSource(),aDir.c_str())),
   mCibl (cSysCoord::FromXML(aCC.SystemeCible(),aDir.c_str()))
{
}
*/

cChSysCo::cChSysCo(cSysCoord * aSrc,cSysCoord * aCibl) :
   mSrc  (aSrc),
   mCibl (aCibl)
{
}

cChSysCo * cChSysCo::Alloc(const std::string & aName,const std::string & aDirGlob)
{
    std::string aSrc,aCibl;
    if ( SplitIn2ArroundCar(aName,'@',aSrc,aCibl,true))
    {
         return new cChSysCo
                    (
                          cSysCoord::FromFile(aDirGlob+aSrc),
                          cSysCoord::FromFile(aDirGlob+aCibl)
                    );
    }
    cChangementCoordonnees aCC= StdGetObjFromFile<cChangementCoordonnees>
                              (
                                  aDirGlob+aName,
                                  StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                  "ChangementCoordonnees",
                                  "ChangementCoordonnees"
                               );

    // return new cChSysCo(aCC,DirOfFile(aName));
    std::string aDir = DirOfFile(aDirGlob+aName);

    return new cChSysCo
               (
                    cSysCoord::FromXML(aCC.SystemeSource(),aDir.c_str()),
                    cSysCoord::FromXML(aCC.SystemeCible(),aDir.c_str())
               );
}

Pt3dr  cChSysCo::Src2Cibl(const Pt3dr & aP) const
{
    return mCibl->FromGeoC(mSrc->ToGeoC(aP));
}
Pt3dr  cChSysCo::Cibl2Src(const Pt3dr & aP) const
{
    return mSrc->FromGeoC(mCibl->ToGeoC(aP));
}
std::vector<Pt3dr>  cChSysCo::Src2Cibl(const std::vector<Pt3dr> & aP) const
{
    return mCibl->FromGeoC(mSrc->ToGeoC(aP));
}
std::vector<Pt3dr>  cChSysCo::Cibl2Src(const std::vector<Pt3dr> & aP) const
{
    return mSrc->FromGeoC(mCibl->ToGeoC(aP));
}


void cChSysCo::ChangCoordCamera(const std::vector<ElCamera *> & aVCam,bool ForceRot)
{
    ElCamera::ChangeSys(aVCam,*this,ForceRot,true);
}

cChSysCo::~cChSysCo()
{
}

/*
*/

/*
        Pt3dr File2GeoC(const Pt2dr & ) const;
*/



/*

        Pt3dr Loc2File(const Pt3dr & ) const;
        Pt3dr Loc2File(const Pt2dr & ) const;  // Valide si ZMoyen
        Pt3dr Loc2GeoC(const Pt3dr & ) const;
        Pt3dr Loc2GeoC(const Pt2dr & ) const;
*/


/*************************************************/
/*                                               */
/*               cTransfo3D                      */
/*                                               */
/*************************************************/

cTransfo3D * cTransfo3D::Alloc(const std::string & aName,const std::string & aDir) 
{
    if (ELISE_fp::exist_file(aDir+aName) && IsPostfixedBy(aName,"xml"))
    {
         cXml_ParamBascRigide  *  aXBR = OptStdGetFromPCP(aDir+aName,Xml_ParamBascRigide);
         if (aXBR)
         {
            cSolBasculeRig * aRes = new cSolBasculeRig(Xml2EL(*aXBR));

            delete aXBR;
            return aRes;
         }
    }
    return  cChSysCo::Alloc(aName,aDir);
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
