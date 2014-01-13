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
#include "bench.h"
#include <algorithm>

void ShowMatr(const char * mes, ElMatrix<REAL> aMatr);
static Pt3dr PRan(REAL Ampl)
{
	return  Pt3dr(NRrandC(),NRrandC(),NRrandC()) * Ampl;
}

static ElMatrix<REAL> RotRand(REAL Ampl)
{
   return ElMatrix<REAL>::Rotation
          (
              NRrandC()*Ampl,
              NRrandC()*Ampl,
              NRrandC()*Ampl
          );
}

// RotP2toP1 : Envoie un point en coord Cam2 vers un point en Coord Cam1
ElRotation3D PerturbRot(ElRotation3D RotP2toP1,REAL AmplTrans,REAL AmplTeta)
{
   Pt3dr aCOpt2 =  RotP2toP1.ImAff(Pt3dr(0,0,0));

   return ElRotation3D
          (
                 aCOpt2 + PRan(euclid(aCOpt2) * AmplTrans),
                 RotP2toP1.Mat() * RotRand(AmplTeta),
		 true // __NEW
          );
}

double EcartRot(const ElRotation3D & R1,const ElRotation3D & R2)
{
   REAL aRes = 0;
   INT aNb = 100;
   for (INT  aK=0 ; aK<aNb ; aK++)
   {

      Pt3dr aP1 = Pt3dr(NRrandC(),NRrandC(),NRrandC()) * 20;
      Pt3dr aP2 = R1.ImAff(aP1);
      Pt3dr aQ2 = R2.ImAff(aP1);

      aRes +=  euclid(aP2-aQ2);
   }
   return aRes / aNb;
}

/**********************************************/

void BenchAmelLocMepRel
    (
         CamStenopeIdeale                aCamInit1,
         std::vector<CamStenopeIdeale *> VCams, 
         std::vector<ElPackHomologue>    VPack   
    )
{
   /*
   bool FocPPFree = INT(VCams.size()) > 1;

   REAL  FocInit =  FocPPFree ? 1.02 : 1.0;
   Pt2dr PPInit  =  FocPPFree ? Pt2dr(0.03,-0.02) : Pt2dr(0,0);

   cSetEqFormelles aSet;

   cParamIntrinsequeFormel * aPIF = aSet.NewParamIntrNoDist(FocInit,PPInit);
//----- comment
   cParamIntrinsequeFormel * aPIF = aSet.NewIntrDistRad
    (FocInit,PPInit,3,ElDistRadiale_PolynImpair(Pt2dr(0,0)));
//-----
   cCameraFormelle * aCam1  = aPIF->NewCam
	                      (
                                  cNameSpaceEqF::eRotFigee,
				  ElRotation3D(Pt3dr(0,0,0),0,0,0)
			      );

    aPIF->SetFocFree(FocPPFree);
    aPIF->SetPPFree(FocPPFree);


   std::vector<cCpleCamFormelle *> VCples;
   for (INT kCam=0; kCam<INT(VCams.size()) ; kCam++)
   {
       // Envoie un point en coord Cam2 vers un point en Coord Cam1
       ElRotation3D RotP2toP1 = aCamInit1.Orient() * (*VCams[kCam]).Orient().inv();

       ElRotation3D aRPert = PerturbRot(RotP2toP1,0.1,0.01);
       cCameraFormelle * aCam2  = aPIF->NewCam(cNameSpaceEqF::eRotBaseU,aRPert,aCam1);


       cCpleCamFormelle * aCpl12 =  aSet.NewCpleCam(*aCam1,*aCam2);
       VCples.push_back(aCpl12);
   }
   // OK

   aSet.SetClosed();

   REAL EcRot = 1e6;
   REAL EcartEq = 1e6;
   for (INT k=0; k< 6 ; k++)
   {
       aSet.AddContrainte(aCam1->RF().StdContraintes());
       aSet.AddContrainte(aPIF->StdContraintes());

       for (INT kCple=0; kCple<INT(VCples.size()) ; kCple++)
           aSet.AddContrainte(VCples[kCple]->Cam2().RF().StdContraintes());

       EcRot=0;
       EcartEq=0;

       for (INT kCple=0; kCple<INT(VCples.size()) ; kCple++)
       {
           ElRotation3D RotP2toP1 = aCamInit1.Orient() * (*VCams[kCple]).Orient().inv();
           ElRotation3D aCur = VCples[kCple]->Cam2().CurRot();
           Pt3dr aTr = vunit(aCur.tr()) * euclid(RotP2toP1.tr());
           aCur = ElRotation3D(aTr,aCur.Mat());

           EcRot += EcartRot(RotP2toP1,aCur);
           EcartEq += VCples[kCple]->AddPackLiaisonP1P2(VPack[kCple],false);

           cout<< "Ecart Eq = " << EcartEq  << " ROT " <<  EcRot << "\n";
       }
//---- comment
       REAL Foc = aPIF->CurFocale();
       Pt2dr PP = aPIF->CurPP();
       cout << Foc << PP << "\n";
//----
       aSet.SolveResetUpdate();
   }
   BENCH_ASSERT(ElAbs(aPIF->CurFocale()-1.0)<epsilon);
   BENCH_ASSERT(euclid( aPIF->CurPP())<epsilon);
   BENCH_ASSERT(EcartEq<epsilon);
   BENCH_ASSERT(EcRot<epsilon);
   cout << "\n\n";
*/
}

/**********************************************/

void ShowRot(const char * mes,ElRotation3D aRot)
{

   cout   << mes << " : " << aRot.tr()
          << " " << aRot.teta01()
          << " " << aRot.teta02()
          << " " << aRot.teta12() << "\n";
}




static ElRotation3D RanRot()
{
	//return ElRotation3D (PRan(5),RotRand(100));
	return ElRotation3D (PRan(5),RotRand(100),true); // __NEW
}





static bool OkCams
            (
	        CamStenopeIdeale & aCam1,
		CamStenopeIdeale & aCam2,
		bool PCoPlan
            )
{
   //Pt3dr aC1 = aCam1.CentreOptique();
   //Pt3dr aC2 = aCam2.CentreOptique();
   Pt3dr aC1 = aCam1.PseudoOpticalCenter(); // __NEW
   Pt3dr aC2 = aCam2.PseudoOpticalCenter(); // __NEW

   if (euclid(aC1-aC2)<3.0)
      return false;

   if (! PCoPlan)
      return true;

   return (ElAbs(aC1.z)>0.1) && (ElAbs(aC2.z)>0.1) ;
}




void OneBenchMatEss(bool CamPhys,bool PCoPlan,bool AmelLoc)
{
   static INT aCPT = 0;
   aCPT++;
   
   //CamStenopeIdeale aCam1(1.0,Pt2dr(0,0));
   //CamStenopeIdeale aCam2(1.0,Pt2dr(0,0));
   //CamStenopeIdeale aCam3(1.0,Pt2dr(0,0));
   //CamStenopeIdeale aCam4(1.0,Pt2dr(0,0));
   vector<double> AFocalParam;
   CamStenopeIdeale aCam1( false /*isDistC2M*/, 1.0, Pt2dr(0,0), AFocalParam ); // __NEW
   CamStenopeIdeale aCam2( false, 		1.0, Pt2dr(0,0), AFocalParam ); // __NEW
   CamStenopeIdeale aCam3( false, 		1.0, Pt2dr(0,0), AFocalParam ); // __NEW
   CamStenopeIdeale aCam4( false, 		1.0, Pt2dr(0,0), AFocalParam ); // __NEW

   while (! (    OkCams(aCam1,aCam2,PCoPlan)
              && OkCams(aCam1,aCam3,PCoPlan)
              && OkCams(aCam1,aCam4,PCoPlan)
            )
         )
   {
       aCam1.SetOrientation( RanRot());
       aCam2.SetOrientation( RanRot());
       aCam3.SetOrientation( RanRot());
       aCam4.SetOrientation( RanRot());

      if (CamPhys)
      {
          //aCam1.SetOrientation(ElRotation3D (PRan(1),RotRand(0.2)));
          //aCam2.SetOrientation(ElRotation3D (PRan(1)+Pt3dr(10,0,0),RotRand(0.2)));
          //aCam3.SetOrientation(ElRotation3D (PRan(1)+Pt3dr(0,10,0),RotRand(0.2)));
          //aCam4.SetOrientation(ElRotation3D (PRan(1)+Pt3dr(-10,0,0),RotRand(0.2)));
          aCam1.SetOrientation( ElRotation3D( PRan(1),               RotRand(0.2), true ) );
          aCam2.SetOrientation( ElRotation3D( PRan(1)+Pt3dr(10,0,0), RotRand(0.2), true ) );
          aCam3.SetOrientation( ElRotation3D( PRan(1)+Pt3dr(0,10,0), RotRand(0.2), true ) );
          aCam4.SetOrientation( ElRotation3D( PRan(1)+Pt3dr(-10,0,0),RotRand(0.2), true ) );
      }
   }

 // aCam1.SetOrientation(ElRotation3D(Pt3dr(0,0,0),0,0,0));
 // aCam2.SetOrientation(ElRotation3D(Pt3dr(1,1,0),1,0.0,0.0));


   std::vector<ElPackHomologue>    VPack;  

   std::vector<CamStenopeIdeale *> VCams; 
   VCams.push_back(&aCam2);
   VCams.push_back(&aCam3);
   VCams.push_back(&aCam4);

   for (INT KCam=0; KCam<INT(VCams.size()) ; KCam++)
   {
       VPack.push_back(ElPackHomologue());
       CamStenopeIdeale & CAM2 = *(VCams[KCam]);
       ElPackHomologue & PACK = VPack[KCam];
       INT NbP = 0;
       while(NbP < 400)
       {
            Pt3dr aP =  Pt3dr(NRrandC(),NRrandC(),NRrandC()) * 20;

	    if (CamPhys) 
            {
               if (PCoPlan)
                  aP.z = 20;
	       else
                  aP.z = 20 + NRrandC() * 5;
	    }
	    else
            {
	       if (PCoPlan)
                  aP.z = 0;
	    }

	    Pt2dr aU1 = aCam1.R3toF2(aP);
	    Pt2dr aU2 = CAM2.R3toF2(aP);


	    if ((euclid(aU1)<10) && (euclid(aU2)<10))
	    {
                INT RatErr = (PCoPlan ? 20 : 4);
           
                if (((NbP % RatErr) == 0)  && (! AmelLoc))
                   aU2 +=  Pt2dr(NRrandC(),NRrandC()) * ((CamPhys || PCoPlan) ? 0.1 :0.5);

                //PACK.add(ElCplePtsHomologues(aU1,aU2));
                PACK.Cple_Add(ElCplePtsHomologues(aU1,aU2)); // __NEW
	        NbP++;
	    }
       }
   }
   ElPackHomologue & aPack = VPack[0];
   if (AmelLoc)
   {
	   BenchAmelLocMepRel(aCam1,VCams,VPack);
	   return;
   }

   std::list<ElRotation3D> lRot;
   ElRotation3D aRot =  aCam2.Orient() *aCam1.Orient().inv();  

   REAL LBase = euclid(aRot.tr());

   if (PCoPlan)
   {
       cResMepRelCoplan aSol = aPack.MepRelCoplan(LBase,false);
       if (CamPhys) 
       {
            cElemMepRelCoplan & bSol = aSol.BestSol();
            ElRotation3D bRot = bSol.Rot();
           lRot.push_back(bRot);

       }
       else
           lRot= aSol.LRot();
   }
   else
   {

       //ElMatrix<REAL> aMEss = aPack.MatriceEssentielle();
       ElMatrix<REAL> aMEss = aPack.MatriceEssentielle(false); // __NEW

   // ElMatrix<REAL> aU(3,3),aDiag(3,3),aV(3,3);
   // svdcmp_diag(aMEss,aU,aDiag,aV);

       for (INT aK= 0 ; aK< 10 ; )
       {
            Pt3dr aP =  Pt3dr(NRrandC(),NRrandC(),NRrandC()) * 20;

	    Pt2dr aU1 = aCam1.R3toF2(aP);
	    Pt2dr aU2 = aCam2.R3toF2(aP);

	    if ((euclid(aU1)<10) && (euclid(aU2)<10))
	    {
	        aK++;
                Pt3dr aR1(aU1.x,aU1.y,1);
                Pt3dr aR2(aU2.x,aU2.y,1);

	        REAL aS = scal(aR1,aMEss*aR2);

	        BENCH_ASSERT(ElAbs(aS)<epsilon);
	    }
       }
       
       if (CamPhys) 
           //lRot.push_back(aPack.MepRelPhysStd(LBase));
           lRot.push_back(aPack.MepRelPhysStd(LBase,false)); // __NEW
       else
           //lRot= aPack.MepRelStd(LBase);
           lRot= aPack.MepRelStd(LBase,false); // __NEW
   }


   REAL dMin = 1e20;
   for 
   (
         std::list<ElRotation3D>::iterator it=lRot.begin(); 
	 it!=lRot.end() ; 
	 it++
   )
   {
	   ElSetMin(dMin,EcartRot(aRot,*it));
   }
   cout << CamPhys << " " << PCoPlan << " dMin = " << dMin << "\n\n";
   BENCH_ASSERT(dMin<epsilon);

   // getchar();
}
 

//   void OneBenchMatEss(bool CamPhys,bool PCoPlan,bool AmelLoc)
void bench_new_mep_rel()
{
     for (INT aK=0; aK<20 ; aK++)
     {
         OneBenchMatEss(true,false,false);
         OneBenchMatEss(false,false,false);
         OneBenchMatEss(true,true,false);
         OneBenchMatEss(false,true,false);

         OneBenchMatEss(true,false,true);
	 // COPLANAIRE , ca foire
         OneBenchMatEss(true,true,true);
     }
}


void Bench_Xml_Image(int aTx,int aTy,Fonc_Num aF)
{
   Im2D_REAL4  anI1(aTx,aTy);
   ELISE_COPY(anI1.all_pts(),aF,anI1.out());

   cElXMLTree * aTree = ToXMLTree("toto",anI1);

   Im2D_REAL4 anI2;
   xml_init(anI2,aTree);
   delete aTree;

   BENCH_ASSERT(anI1.tx()==anI2.tx());
   BENCH_ASSERT(anI1.ty()==anI2.ty());
   double aMaxDif;
   ELISE_COPY(anI1.all_pts(),Abs(anI1.in()-anI2.in()),VMax(aMaxDif));
   BENCH_ASSERT(aMaxDif==0.0);
}

void Bench_Xml_Image()
{
   Bench_Xml_Image(1,1,3.4567);
   Bench_Xml_Image(10,14,FX+FY);
   for (int aK=0 ; aK< 20 ; aK++)
   {
       Bench_Xml_Image(1+NRrandom3(10),1+NRrandom3(10),frandr());
   }
   std::cout << "OK Bench_Xml_Image \n";
}

void bench_xml()
{
     Bench_Xml_Image();

     ElDistRadiale_PolynImpair aDist(1000,Pt2dr(1,2));
     aDist.PushCoeff(1e-3);
     aDist.PushCoeff(-1e-5);
     aDist.PushCoeff(1e-7);
     aDist.PushCoeff(1e-9);
     //cCamStenopeDistRadPol aCam(0.123e-3,Pt2dr(5,6),aDist);
     cCamStenopeDistRadPol aCam( false /*isDistC2M*/, 0.123e-3,Pt2dr(5,6),aDist, vector<double>() ); // __NEW

     {
        cElXMLFileIn aFileXML("BENCH_FILE_IM/T.xml");
        aFileXML.PutCam(aCam);
     }

     /*
     FILE * fp = fopen("BENCH_FILE_IM/T.xml","r");
     while (1)
     {
        cElXMLToken aTok(fp);
	cout << "[" << aTok.Val() << "]" << INT(aTok.Kind()) << "\n";
	if (aTok.Kind()==eXMLEOF)
           return;
     }
     */

     cElXMLTree aTree("BENCH_FILE_IM/T.xml");
     aTree.Show("  ",stdout,0,0,true);
}



