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


/*
*/

#include "all_etal.h"
#include <algorithm>

#include "XML_GEN/all.h"
using namespace NS_ParamChantierPhotogram;



static void SauvDradDtx
     (
          FILE * aFTxt,
          const std::string & aName,
          ElDistRadiale_PolynImpair & aDist,
	  REAL aMaxRay =-1
      )
{

     fprintf(aFTxt,"%s ",aName.c_str());
     for (INT aK =0 ; aK<aDist.NbCoeff() ; aK++)
     {
         REAL aC = aDist.Coeff(aK);
         fprintf(aFTxt,"%e ",aC);
     }
     fprintf(aFTxt,"\n");

     if (aMaxRay > 0)
     {
        int aNbR = 20;
        for (int aK=0 ; aK<=aNbR ; aK++)
	{

	     double aR =  (aMaxRay*aK) / aNbR;
	     fprintf(aFTxt,"Dist(%lf) = %lf\n", aR,aR*aDist.DistDirecte(aR));
	}
    }

}

void  cEtalonnage::SauvLeica(CamStenope & aCamElise)
{
    double aResol = mParam.TaillePixelExportLeica();
    if (aResol <= 0)
       return;

   cCamStenopeDistRadPol *aCLeica = aCamElise.Change2Format_DRP
                                    (
				         false,
					 2,
					 true,
					 aResol,
					 Pt2dr(mSzIm)/2.0
				    );
   ElDistRadiale_PolynImpair aDist = aCLeica->DRad();

   std::string aNF = mParam.Directory() + mParam.NameCamera() + ".cam";
   FILE * aFP = ElFopen(aNF.c_str(),"w");
   if (aFP==0)
   {
       std::cout << "Name=" << aNF << "\n";
       ELISE_ASSERT(false,"Cannot open file");
   }

   fprintf(aFP,"%s\n\n", mParam.NameCamera().c_str());
   fprintf(aFP,"%e\n", aCLeica->Focale());
   fprintf(aFP,"%e\n", aCLeica->PP().x);
   fprintf(aFP,"%e\n", aCLeica->PP().y);
   fprintf(aFP,"%e\n", 0.0);
   fprintf(aFP,"%e\n",aDist.Coeff(0));
   fprintf(aFP,"%e\n",aDist.Coeff(1));

    fprintf(aFP,"0\n8\n30 0\n-30 0\n0 30\n0 -30\n2\n0 0\n");

   std::cout << mParam.NameCamera() << "\n";
   std::cout << aCLeica->Focale() <<  aCLeica->PP() << "\n";
   std::cout <<aDist.Centre() << "\n";
   for (int aK=0 ; aK<5 ; aK++)
       std::cout << aDist.Coeff(aK) << "\n";

   ElFclose(aFP);
}

void cEtalonnage::SauvXML(const std::string & aName)
{
    CamStenope  * aCam = mParamIFGen->CurPIF();
    cCalibrationInternConique aCIC = aCam->ExportCalibInterne2XmlStruct(Pt2di(mSzIm));
    MakeFileXML(aCIC,mDir+aName+mParam.NameCamera() + ".xml");

}

static std::vector<double> NoParAdd;

void cEtalonnage::SauvDRad(const std::string & aName,const std::string & aNamePhgrStd)
{
     bool C2M = mParam.ModeC2M();
     std::string aNameTxt = mDir + aName + ".txt";
     std::string aNameBin = mDir + aName + ".dat";

     ELISE_fp aFileBin(aNameBin.c_str(),ELISE_fp::WRITE);
     FILE * aFTxt = ElFopen(aNameTxt.c_str(),"w");

     Pt2dr PP = CurPP();
     REAL  F  = CurFoc();

     ElDistRadiale_PolynImpair aDist = CurDist();



     fprintf(aFTxt,"PP = %lf %lf %lf\n",PP.x,PP.y,F);
     fprintf(aFTxt,"CDIST = %lf %lf\n",aDist.Centre().x,aDist.Centre().y);


     ElDistRadiale_PolynImpair aDInv = aDist.DistRadialeInverse(mMaxRay,-2);

     SauvDradDtx(aFTxt,"Cam to monde D3 ",C2M ?aDist:aDInv);
     SauvDradDtx(aFTxt,"Monde to Cam D3 ",C2M ?aDInv:aDist,mMaxRay);

     {
         
         cCamStenopeDistRadPol aCDRP(mParam.ModeC2M(),F,PP,aDist,NoParAdd);
	 aCDRP.SetSz(Pt2di(mSzIm));
	 // aCDRP.SetDistInverse();
	 // cCalibrationInternConique aCIC = aCDRP.ExportCalibInterne2XMLStruct(mSzIm);
	  cCalibrationInternConique aCIC = aCDRP.ExportCalibInterne2XmlStruct(Pt2di(mSzIm));
	   MakeFileXML(aCIC,mDir+aName+mParam.NameCamera() + ".xml");

         // SauvXML(aName);
	 SauvLeica(aCDRP);

         //
         cDistModStdPhpgr aDP(aDist);
         cCamStenopeModStdPhpgr aCP(mParam.ModeC2M(),F,PP,aDP,NoParAdd);
	  aCIC = aCP.ExportCalibInterne2XmlStruct(Pt2di(mSzIm));
	   MakeFileXML(aCIC,mDir+aNamePhgrStd+mParam.NameCamera() + ".xml");
     }
     // aDInv = aDist.DistRadialeInverse(mMaxRay,0);
     // SauvDradDtx(aFTxt,"Mond e to Cam D5 ",aDInv);
     ElFclose(aFTxt);


     aFileBin.write(F);
     aFileBin.write(PP);
     aDist.write(aFileBin);

     aFileBin.close();
}

class   cEtDistFinale : public ElDistortion22_Gen
{
	public :
	  cEtDistFinale(cEtalonnage & anEtal,CamStenope * pCam) :
               mEtal (anEtal),
               mCam  (*pCam)
	  {
	  }

          Pt2dr Direct(Pt2dr aP) const  
	  {
	       return mCam.F2toPtDirRayonL3(mEtal.ToPN(aP));
	  }
        private :
          bool OwnInverse(Pt2dr & aP) const 
	  {
              aP = mEtal.FromPN(mCam.PtDirRayonL3toF2(aP));
              return true;
          }
          void  Diff(ElMatrix<REAL> &,Pt2dr) const 
	  {
              ELISE_ASSERT(false,"cEtDistFinale::Diff");
	  }

          cEtalonnage mEtal;
          CamStenope & mCam;
};



void cEtalonnage::SauvDataGrid(Im2D_REAL8 anIm,const std::string & aName)
{
      std::string aFullName = mParam.Directory() + aName + ".dat";
     ELISE_fp  aFp(aFullName.c_str(),ELISE_fp::WRITE);
     aFp.write(anIm.data_lin(),sizeof(double),anIm.tx()*anIm.ty());
     aFp.close();
}


void cEtalonnage::SauvGrid(REAL aStep,const std::string & aName,bool XML)
{
     ElTimer aChrono;
     cout << "BEGIN = " << aChrono.uval() << "\n";

     CamStenope * aCam =  mParamIFGen->CurPIF();

     ElDistRadiale_PolynImpair *  aDRP = static_cast<ElCamera *>(aCam)->Get_dist().DRADPol();
     if (aDRP)
     {
	REAL aRCr = aDRP->RMaxCroissant(mMaxRay*20);
        Pt2dr aC = aDRP->Centre();

	REAL aD=0.0;
	Box2dr aBox(ToPN(Pt2dr(0,0)),ToPN(mSzIm));
	Pt2dr aCorners[4];
	aBox.Corners(aCorners);
	for (INT aK=0 ; aK<4 ; aK++)
            ElSetMax(aD,euclid(aC,aCorners[aK]));

	ELISE_ASSERT(aD<aRCr,"Polynome non inversible sur le domaine");
	aDRP->SetRMax(ElMin(aRCr,aD*1.1));


     }

    
     cEtDistFinale anEDF(*this,aCam);


     std::string aNameGr =  aName + mParam.NameCamera();

      Pt2di aRab = mParam.RabExportGrid();
      cDbleGrid aGr(true,Pt2dr(0,0)-Pt2dr(aRab),mSzIm+Pt2dr(aRab),Pt2dr(aStep,aStep),anEDF,aNameGr);
// cDbleGrid aGr(Pt2dr(0,0),mSzIm,aStep,anEDF,aNameGr);
     cout << "Time = " << aChrono.uval() << "\n";
     cout << "PHGR : " << aGr.Focale() << " " << aGr.PP() << "\n";
     if (mParamIFDR !=0) 
        cout << "DRAD  : " << CurFoc() -aGr.Focale()  
             << " " << CurPP() -  aGr.PP()<< "\n";


      
     if (XML)
     {
         std::string aNameThom = mParam.ANameFileExistant();
	 ThomParam  aThomP(aNameThom.c_str());
         std::string aNameCapt = mParam.Directory() + std::string("capteur.xml");
         cElXMLFileIn aFileXml(mParam.Directory()+aNameGr+"_MetaDonnees.xml");

         ElDistRadiale_PolynImpair * aPtrDistRad = 0;
         ElDistRadiale_PolynImpair aPol(0,Pt2dr(0,0));
         Pt2dr aPP,* aPtrPP=0;
         double aFoc,*aPtrFoc = 0;
         if (mParamIFDR)
         {
            aPol = CurDistInv(-2);
            aPP=CurPP();
            aFoc = CurFoc();
            aPtrDistRad = &aPol;
            aPtrPP = & aPP;
            aPtrFoc = & aFoc;

         }
         

         aFileXml.SensorPutDbleGrid
         (
	      Pt2di(mSzIm),
	      mParam.XMLAutonome(),
              aGr,
              aNameThom.c_str(),
              // aThomP.BIDON ? 0 : aNameCapt.c_str(),
               (! ELISE_fp::exist_file( aNameCapt))  ? 0 : aNameCapt.c_str(),
              aPtrDistRad,
              aPtrPP,
              aPtrFoc
         );
	 
	 if (! mParam.XMLAutonome())
	 {
	    SauvDataGrid(aGr.GrDir().DataGridX(),aGr.GrDir().NameX());
	    SauvDataGrid(aGr.GrDir().DataGridY(),aGr.GrDir().NameY());
	    SauvDataGrid(aGr.GrInv().DataGridX(),aGr.GrInv().NameX());
	    SauvDataGrid(aGr.GrInv().DataGridY(),aGr.GrInv().NameY());
	 }
     }
     else
         aGr.write(mDir+aName+std::string(".Grid"));

    SauvPointes
    (
          mParam.Directory() + "Pointes" +mModeDist+".pk1",
         &aGr
    );

}

void cEtalonnage::SauvPointes(const std::string & aName,cDbleGrid * aGr)
{
      FILE * fp = ElFopen(aName.c_str(),"w");
      ELISE_ASSERT(fp!=0,"cEtalonnage::SauvPointes ElFopen");

      for (tContCam::iterator itC=mCams.begin() ; itC!=mCams.end() ; itC++)
      {
            cCamIncEtalonage * pCam = *itC;
            fprintf(fp,"%s\n",pCam->Name().c_str());
                                                                                
            cSetPointes1Im::tCont & aSet = pCam->SetPointes().Pointes();
            for
            (
               cSetPointes1Im::tCont::iterator itS=aSet.begin();
               itS!=aSet.end();
               itS++
            )
            {
                Pt2dr aP = FromPN(itS->PosIm());
                if (aGr)
                {
                      aP = aGr->Direct(aP);
                      aP = mNormId->ToCoordIm(aP);
                }
                fprintf(fp,"%d %lf %lf\n",itS->Cible().Ind(),aP.x,aP.y);
            }
            fprintf(fp,"-1\n");
      }
      fprintf(fp,"-1\n");
      ElFclose(fp);
}


void  cEtalonnage::ExportAsDico(const cExportAppuisAsDico& anExp)
{
    cDicoAppuisFlottant aDic;

    // aDic.NameDico() = anExp.NameDico();
    for 
    (
         cPolygoneEtal::tContCible::const_iterator itC= mPol.ListeCible().begin();
         itC!= mPol.ListeCible().end();
	 itC++
    )
    {
        cOneAppuisDAF anAp;  
	anAp.Pt() =  (*itC)->Pos();
	anAp.NamePt() = ToString((*itC)->Ind());
	anAp.Incertitude() = anExp.Incertitude();

	aDic.OneAppuisDAF().push_back(anAp);
    }

    MakeFileXML
    (
        aDic,
	  mParam.Directory()
	+ std::string("DicoOfApp_") 
	+ anExp.NameDico() 
	+ std::string(".xml")
    );

    for (tContCam::iterator itC=mCams.begin() ; itC!=mCams.end() ; itC++)
    {
        (*itC)->ExportAsDico(anExp);
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
