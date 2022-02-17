/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/


#include "StdAfx.h"


class GMCube
{
   public :
     GMCube(std::string);

      Pt2di mP0;
      Pt2di mSz;
      // INT   mZ0; INT   mZ0;


      ELISE_fp aFp;

      REAL mX0; 
      REAL mY0; 
      REAL mZ0; 

      REAL mStepXY;
      REAL mStepZ;

      INT mTx; 
      INT mTy; 
      INT mTz; 

      Im2D_INT2 mIZmin;
      Im2D_INT2 mIZmax;
      REAL      mZoom;
      Video_Win mW;
      Video_Win mW2;
      Video_Win mW3;
      Video_Win mW4;
      Video_Display  mDisp;


      cInterfaceCoxAlgo * pSCA;

      void Opt();

      ~GMCube();
};


GMCube::~GMCube()
{
   delete pSCA;
}

void ShowIm2(Im2D_INT2 aZ,Video_Win aW)
{
    INT zMax,zMin;

    ELISE_COPY(aZ.all_pts(),aZ.in(),VMax(zMax)|VMin(zMin));
    cout << "Z  = [" << zMin << " , " << zMax << "]\n";

    ELISE_COPY
    (
        aZ.all_pts(),
        Max(0,Min(255,aZ.in())), // (aZ.in()-zMin) * (255.0/(zMax-zMin)),
        aW.ogray()
    );
}

void GMCube::Opt()
{ 
    REAL aCapaTot = 0;
    ElTimer aTimer;
    INT aCPT = 0;
    {
// if ((aCPT %10==0) || (aCPT>=155)) pSCA->NbChem();
           INT aCapa = pSCA->PccMaxFlow();
           aCapaTot += aCapa;

	   aCPT++;
           if ((aCPT%1 == 0) || (aCapa == 0))
                cout << "CPT " << aCPT 
		<< " Time : " << aTimer.uval() 
		<< " DCapa : " << aCapa 
                << " Som Capa = " << aCapaTot << "\n";



           {
cout << "AAAAAAAAAAA\n";
              Im2D_INT2 aZ = pSCA->Sol(0);

              Tiff_Im  aTif
                       (
                           "/home/pierrot/Data/Cox.tif",
                           mSz,
                           GenIm::u_int1,
                           Tiff_Im::No_Compr,
                           Tiff_Im::BlackIsZero 
                       );
              ELISE_COPY(aZ.all_pts(),aZ.in(),aTif.out());
              
	      ShowIm2(aZ,mW4);
cout << "BBBBBBBB\n";
	      return;
           }
    }
    
}

GMCube::GMCube(std::string aName)  :
    mP0     (0,0),
    mSz     (20000,20000),
    aFp     ( aName.c_str(),ELISE_fp::READ),
    mX0     ( aFp.read_REAL8()),
    mY0     ( aFp.read_REAL8()),
    mZ0     ( aFp.read_REAL8()),
    mStepXY ( aFp.read_REAL8()),
    mStepZ  ( aFp.read_REAL8()),
    mTx     ( aFp.read_INT4()),
    mTy     ( aFp.read_INT4()),
    mTz     ( aFp.read_INT4()),
    mIZmin  ( mTx, mTy),
    mIZmax  ( mTx, mTy),
    mZoom   (1.0),
    mW      (Video_Win::WStd(Pt2di(mTx,mTy),mZoom)),
    mW2     (Video_Win::WStd(Pt2di(mTx,mTy),mZoom)),
    mW3     (Video_Win::WStd(Pt2di(200,100),mZoom)),
    mW4     (Video_Win::WStd(Pt2di(mTx,mTy),mZoom)),
    mDisp   (mW2.disp())
{

   mSz = Inf(mSz,Pt2di(mTx,mTy));


   INT aNbZ =0;
   long aDeb = aFp.tell();
   INT aMaxZ =0;

   INT aVMax = -1000000;
   INT aVMin = 100000;
   for(INT y=0;y<mTy;++y)
   {

      for(INT x=0;x<mTx;++x)
      {
          INT2 zM  = aFp.read_INT2();
          INT2 aN  = aFp.read_INT2();

          mIZmin.data()[y][x] = zM;
          mIZmax.data()[y][x] = zM+aN;
          ElSetMax(aMaxZ ,(INT) aN);

          for (INT z=0 ; z<aN ; z++)
          {
              INT2 aVal =  aFp.read_INT2();
              ElSetMin(aVMin,aVal);
              ElSetMax(aVMax,aVal);
          }
          aNbZ += aN;
      }
   }

   cout << "NbZ = " << aNbZ <<  "  " << mStepZ << "\n";

   pSCA =     cInterfaceCoxAlgo::StdNewOne
              (
                  mSz,
                  trans(mIZmin.in(),mP0),
                  trans(mIZmax.in(),mP0),
                  2,
                  true
              );
cout << "AAAAAAAAAAAAa\n";

   
   aFp.seek_begin(aDeb);
   Im2D_INT2 zCor(mTx,mTy);
   Im2D_U_INT1 CorMax(mTx,mTy);


   for(INT y=0;y<mTy;++y)
   {
      for(INT x=0;x<mTx;++x)
      {
          INT2 z1  = aFp.read_INT2();
          INT2 z2  = z1+aFp.read_INT2();
 
          INT cMax = -100000;
          INT zMax = z1;

          bool OkP2 = (x>=mP0.x) && (x<mP0.x+mSz.x) && (y>=mP0.y) && (y<mP0.y+mSz.y);
          for (INT z=z1 ; z<z2 ; z++)
          {
              INT2 aVal =  aFp.read_INT2();
              if (aVal > cMax)
              {
                   cMax = aVal;
                   zMax = z;
              }
              aVal = round_ni((aVMax-aVal)/(aVMax/100.0));
              if (OkP2)
                  pSCA->SetCost(Pt3di(x-mP0.x,y-mP0.y,z),aVal);
          }
          zCor.data()[y][x] = zMax;
          CorMax.data()[y][x] = (INT)(255 * (1.0-cMax/900.0));
      }
   }

   cout << "VALS = [" << aVMin << " , " << aVMax << "]\n";

   ELISE_COPY(zCor.all_pts(),CorMax.in(),mW.ogray());
   ELISE_COPY(zCor.all_pts(),zCor.in(),mW2.ogray());

}

REAL test_cox(INT tx,INT ty,INT tz)
{
   ElTimer aT;
   cInterfaceCoxAlgo * pSCA =     cInterfaceCoxAlgo::StdNewOne
              (
                  Pt2di(tx,ty),
                  0,
                  tz,
                  2
              );
   for (INT x=0; x<tx ; x++)
       for (INT y=0; y<ty ; y++)
           for (INT z=0; z<tz ; z++)
                  pSCA->SetCost(Pt3di(x,y,z),INT(100*NRrandom3()));
    pSCA->PccMaxFlow();

    REAL  aRes =  aT.uval();
    cout 
	    << "tCox : "
            << aRes  << " ; "
	    << tx << " " 
	    << ty << " " 
	    << tz << " " 
	    << (aRes *1e4 / (tx*ty*tz)) << "\n";
    delete pSCA;
    return aRes;
}

void bench_cox()
{

  GMCube aGMC("/home/pierrot/Data/Cub/C40/out.cube");
  aGMC.Opt();
 while (1)getchar();

  // GMCube aGMC("/home/pierrot/Data/Cub/out.cube");

   test_cox(10,10,10);
   test_cox(20,10,10);
   test_cox(10,20,10);


   for (INT k= 20 ; k<500 ; k+=10) 
        test_cox(k,k,40);

   for (INT k= 10 ; k<200 ; k+=5) 
        test_cox(30,30,k);




    while (1) getchar();
}






