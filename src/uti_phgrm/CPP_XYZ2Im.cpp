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

/*
*/


cMesureAppuiFlottant1Im *  GetMAFOfNameIm(cSetOfMesureAppuisFlottants & aSMAF,const std::string aNameIm,bool CreatIfNone)
{
   for 
   (
         std::list<cMesureAppuiFlottant1Im>::iterator itMAF=aSMAF.MesureAppuiFlottant1Im().begin();
         itMAF!=aSMAF.MesureAppuiFlottant1Im().end();
         itMAF++
   )
   {
        if (itMAF->NameIm() == aNameIm)
           return &(*itMAF);
   }
   if (CreatIfNone)
   {
       cMesureAppuiFlottant1Im aMAF;
       aMAF.NameIm() = aNameIm;
       aSMAF.MesureAppuiFlottant1Im().push_back(aMAF);
       cMesureAppuiFlottant1Im * aRes = GetMAFOfNameIm(aSMAF,aNameIm,false);
       ELISE_ASSERT(aRes,"Incoherence in GetMAFOfNameIm");
       return aRes;
   }
   return 0;
}



int TransfoCam_main(int argc,char ** argv,bool Ter2Im)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string  aFullNC,aFilePtsIn,aFilePtsOut,aFilteredInput;
    std::string XYZ = "X,Y,Z (xml 3d in Homol mode)";
    std::string IJ = "I,J  (xml 2d  in Homol mode)";
    bool aPoinIsImRef = true;
    bool aInputImWithZ = false;
    std::vector<std::string>  anOptPtH;

    if (Ter2Im)
    {
       ElInitArgMain
       (
           argc,argv,
           LArgMain()  << EAMC(aFullNC,"Nuage or Cam", eSAM_IsExistFile)
                       << EAMC(aFilePtsIn,"File In : " + (Ter2Im ? XYZ : IJ), eSAM_IsExistFile)
                       << EAMC(aFilePtsOut,"File Out : " + (Ter2Im ? IJ : XYZ), eSAM_IsOutputFile),
           LArgMain()
       );
    }
    else
    {
       ElInitArgMain
       (
           argc,argv,
           LArgMain()  << EAMC(aFullNC,"Nuage or Cam", eSAM_IsExistFile)
                       << EAMC(aFilePtsIn,"File In : " + (Ter2Im ? XYZ : IJ), eSAM_IsExistFile)
                       << EAMC(aFilePtsOut,"File Out : " + (Ter2Im ? IJ : XYZ), eSAM_IsOutputFile),
           LArgMain()  << EAM(aFilteredInput,"FilterInput",true,"To generate a file of input superposable to output",eSAM_IsOutputFile)
                       << EAM(aPoinIsImRef,"PointIsImRef",true,"Point must be corrected from cloud resolution def = true")
		       << EAM(aInputImWithZ,"InputImWithZ",false,"Input Im point with Z (for Im2XYZ) def=false")
		       << EAM(anOptPtH,"PtHom",true,"Option for hom =[SH,Im1,Im2]")
       );
    }

    if (!MMVisualMode)
    {
        if (!EAMIsInit(&aFilteredInput))
        {
          aFilteredInput = DirOfFile(aFilePtsIn) + "Filtered_" + NameWithoutDir(aFilePtsIn);
        }


        std::string aDir,aNC;

        SplitDirAndFile(aDir,aNC,aFullNC);

        cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
        cResulMSO aRMso =  anICNM->MakeStdOrient(aNC,false);

	
        cElNuage3DMaille *  aNuage = aRMso.Nuage();
        ElCamera         * aCam    = aRMso.Cam();
	
        bool UseHom = false;
        cDicoAppuisFlottant * aDAF=0;
        cSetOfMesureAppuisFlottants * aSMAF = 0;
        cMesureAppuiFlottant1Im *     aMAF  =0;
        std::string aNameXMLTer="";
        std::string aNameXMLIm="";

        if (EAMIsInit(&anOptPtH))
        {
            ELISE_ASSERT(aNuage,"No Nuage in homol mode");
            aNameXMLIm = aFilePtsIn ;
            aNameXMLTer = aFilePtsOut ;

            ELISE_ASSERT(anOptPtH.size()==3,"Bad size i  hom option");
            std::string aSH=anOptPtH[0],aNameI1=anOptPtH[1],aNameI2=anOptPtH[2];
            
            aFilePtsIn = anICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ aSH+"@txt",aNameI1,aNameI2,true);
            UseHom = true;
            if (! ELISE_fp::exist_file(aFilePtsIn))
            {
                 std::string aBinFilePtsIn = anICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ aSH+"@dat",aNameI1,aNameI2,true);
                 ELISE_ASSERT(ELISE_fp::exist_file(aBinFilePtsIn),"Nor txt nor dat file for homologous option");
                 ElPackHomologue aPack = ElPackHomologue::FromFile(aBinFilePtsIn);
                 aPack.StdPutInFile(aFilePtsIn);
            }
            aSMAF = OptStdGetFromPCP(aNameXMLIm,SetOfMesureAppuisFlottants);
            if (aSMAF==0)
            {
                 aSMAF = new cSetOfMesureAppuisFlottants;
            }
            aMAF =  GetMAFOfNameIm(*aSMAF,aNameI2,true);
            aDAF= OptStdGetFromPCP(aNameXMLTer,DicoAppuisFlottant);
            if (aDAF==0)
            {
                 aDAF = new cDicoAppuisFlottant;
            }
        }


        if (! Ter2Im)
        {
            if ( (aNuage==NULL) && !aInputImWithZ)
            {
                std::cout  << "For name " << aFullNC << "\n";
                ELISE_ASSERT(aNuage!=0,"Is not a MicMac Cloud -XML specif");
            }
        }

	
        ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);
        FILE *  aFOut =  UseHom ? 0 : FopenNN(aFilePtsOut.c_str(),"w","XYZ2Im");
	
        char * aLine;
        std::vector<Pt2dr> aV2Ok;
        bool HasEmpty = false;

        while ((aLine = aFIn.std_fgets()))
        {
            if (Ter2Im)
            {
                Pt3dr aP;
                int aNb = sscanf(aLine,"%lf %lf %lf",&aP.x,&aP.y,&aP.z);
                ELISE_ASSERT(aNb==3,"Could not read 3 double values");

                Pt2dr aPIm;
                if (aNuage) aPIm = aNuage->Terrain2Index(aP);
                if (aCam)   aPIm = aCam->R3toF2(aP);

                fprintf(aFOut,"%lf %lf\n",aPIm.x,aPIm.y);
            }
            else if (UseHom)
            {
                Pt2dr aP1,aP2;
                int aNb = sscanf(aLine,"%lf %lf %lf %lf",&aP1.x,&aP1.y,&aP2.x,&aP2.y);
                ELISE_ASSERT(aNb==4,"Could not read 4 double values");
                 
                if (aPoinIsImRef)
                   aP1 = aNuage->ImRef2Capteur (aP1);
       		if (aNuage->CaptHasData(aP1))
                {
                   std::string aNamePt = "P" + ToString(int(aMAF->OneMesureAF1I().size()));

                   cOneAppuisDAF anAp;
                   anAp.Pt()   = aNuage->PreciseCapteur2Terrain(aP1);
                   anAp.NamePt()   = aNamePt;
                   anAp.Incertitude()   = Pt3dr(1,1,1);
                   aDAF->OneAppuisDAF().push_back(anAp);

// std::cout << "Daaaff " <<  aDAF->OneAppuisDAF().size() << " " << aFilePtsIn << "\n";
                   cOneMesureAF1I aMesIm;
                   aMesIm.PtIm() = aP2;
                   aMesIm.NamePt() = aNamePt;
                   aMAF->OneMesureAF1I().push_back(aMesIm);
                }
            }
            else
            {
                Pt2dr aPIm;
		double aInputZ;
                int aNb = sscanf(aLine,"%lf %lf",&aPIm.x,&aPIm.y);
                ELISE_ASSERT(aNb==2,"Could not read 2 double values");
		if (aInputImWithZ)
		{
			int aNb = sscanf(aLine,"%lf %lf %lf",&aPIm.x,&aPIm.y,&aInputZ);
                	ELISE_ASSERT(aNb==3,"Could not read 3 double values");
			// std::cout << "2D Point + Z: ["<<aPIm.x<< ","<<aPIm.y<<","<<aInputZ<<"]\n";
		}
		else
		{
			int aNb = sscanf(aLine,"%lf %lf",&aPIm.x,&aPIm.y);
                	ELISE_ASSERT(aNb==2,"Could not read 2 double values");
			// std::cout << "2D Point: ["<<aPIm.x<< ","<<aPIm.y<<"]\n";
		}
		if (aNuage)
		{
                        Pt2dr aPIm0 = aPIm;
			if (aPoinIsImRef)
                    	    aPIm = aNuage->ImRef2Capteur (aPIm);/* ici il y a un bug sous linux, segmention core dumped*/

       			if (aNuage->CaptHasData(aPIm))
                	{
                   		Pt3dr aP  = aNuage->PreciseCapteur2Terrain(aPIm);
                   		fprintf(aFOut,"%lf %lf %f\n",aP.x,aP.y,aP.z);
                   		aV2Ok.push_back(aPIm0);
                	}
                	else
                	{
                    		HasEmpty = true;
                    		std::cout << "Warn :: " << aPIm0 << " has no data in cloud\n";
                	}
		}
		else if (aCam)
		{
			Pt3dr aP = aCam->F2AndZtoR3(aPIm,aInputZ);
                	fprintf(aFOut,"%lf %lf %lf\n",aP.x,aP.y,aP.z);
		}
            }
         }

         if (HasEmpty || EAMIsInit(&aFilteredInput))
         {
             FILE *  aFFilter = FopenNN(aFilteredInput.c_str(),"w","XYZ2Im");
             for (int aKP=0 ; aKP<int(aV2Ok.size()) ; aKP++)
             {
                fprintf(aFFilter,"%lf %lf\n",aV2Ok[aKP].x,aV2Ok[aKP].y);
             }
             ElFclose(aFFilter);
         }

        aFIn.close();
        if (UseHom)
        {
            MakeFileXML(*aDAF , aNameXMLTer);
            MakeFileXML(*aSMAF, aNameXMLIm);
        }
        else
        {
           ElFclose(aFOut);
        }

        return 0;
    }
    else
        return EXIT_SUCCESS;
}


int XYZ2Im_main(int argc,char ** argv)
{
    return TransfoCam_main(argc,argv,true);
}

int Im2XYZ_main(int argc,char ** argv)
{
    return TransfoCam_main(argc,argv,false);
}

ElRotation3D ListRot2RotPhys(const  std::list<ElRotation3D> & aLRot,const ElPackHomologue & aPack);
std::list<ElRotation3D>  MatEssToMulipleRot(const  ElMatrix<REAL> & aMEss,double LBase);

void Sho33( ElMatrix<REAL> aM)
{
    for (int aY=0 ; aY<3 ; aY++)
      printf("  %f \t %f \t %f \n",aM(0,aY),aM(1,aY),aM(2,aY));
}

bool ShowMSG_ListRot2RotPhys=false;

//int  PPMD_CalcEss(int argc,char ** argv)
int  PPMD_MatEss2Orient(int argc,char ** argv)
{
   ShowMSG_ListRot2RotPhys = true;

   Pt3dr aRefAll = vunit(Pt3dr(0.0743780117532993751,1.39204657194425052, -0.335883991159797279));
   Pt3dr aRef2 = vunit(Pt3dr(0.08730003023518626, 1.6357168231794974 ,-0.393890910854981069));

   std::string I1,I2,Cal;
   std::string PostH="txt";
   std::string SH="_ConvOLDFormat";
   
   ElInitArgMain
   (
           argc,argv,
           LArgMain()  << EAMC(I1,"Name Im1")
                       << EAMC(I2,"Name Im1")
                       << EAMC(Cal,"Name Calib"),
           LArgMain()
   );

   cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
   std::string   aFilePtsIn = anICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ SH+"@"+PostH,I1,I2,true);

   ElPackHomologue  aPackInit =  ElPackHomologue::FromFile(aFilePtsIn);
   CamStenope *   aCam1 = anICNM->StdCamStenOfNames(I1,Cal);
   CamStenope *   aCam2 = anICNM->StdCamStenOfNames(I2,Cal);

   ElPackHomologue  aPackPh = aCam1->F2toPtDirRayonL3(aPackInit,aCam2);

   std::vector<double> aVMatPy({ -3.48751824,  20.52539487,-361.5038316,  101.79664035,  -6.19477011,
  -18.78353249,  351.48896237,  -22.88157752,1});

   ElMatrix<REAL>  aMatPy(3,3);
   for (int aX=0 ; aX<3 ; aX++)
   {
      for (int aY=0 ; aY<3 ; aY++)
      {
          aMatPy(aX,aY) = aVMatPy.at(aX+3*aY);
      }
   }
/*
 [[  -3.48751824   20.52539487 -361.5038316 ]
 [ 101.79664035   -6.19477011  -18.78353249]
 [ 351.48896237  -22.88157752   1.        ]]

*/
   ElMatrix<REAL> aMM_MEss = aPackPh.MatriceEssentielle(true);
   aMM_MEss *= 1/aMM_MEss(2,2);

   ElRotation3D aR1toW(Pt3dr(0,0,0),0,0,PI);
   std::cout << "==== aR1ToMonde === \n";
   Sho33(aR1toW.Mat());

  
   for (int aKMat=0 ; aKMat<1 ; aKMat++)
   {
      std::cout << "===============\n";
      bool PyMath = (aKMat==0);
      ElMatrix<REAL> aMEss = PyMath ? aMatPy : aMM_MEss;
      Sho33(aMEss);
    
      std::list<ElRotation3D>  aLR = MatEssToMulipleRot(aMEss,1.0);
      for (const auto & aR : aLR)
      {
           std::cout << "=====TR=" << aR.inv().tr() << "\n";
           Sho33( aR.inv().Mat());
      }

      // Orientation Monde to Cam de la camera 1
      ElRotation3D  aR1to2 = ListRot2RotPhys(aLR,aPackPh);
      ElRotation3D  aR2toW = aR1toW * aR1to2.inv();



      std::cout << "WINN = " << aR2toW.tr() << "\n";
      Sho33(aR2toW.Mat());
   }
   if (0)
   {
       std::cout << "Refpts " << aRef2  << " " << aRefAll << "\n";

       ElMatrix<REAL> aSvd1(3,3),aDiag(3,3),aSvd2(3,3);
       svdcmp_diag(aMatPy,aSvd1,aDiag,aSvd2,true);
       std::cout << " ================  SVD1 ============\n";
       Sho33(aSvd1);
       std::cout << " ================  DIAG ============\n";
       Sho33(aDiag);
       std::cout << " ================  SVD2 ============\n";
       Sho33(aSvd2);
   }
   {
       ElMatrix<REAL> MTest =  ElMatrix<REAL>::Rotation(0,0,PI/2);
       Sho33(MTest);
   }


   return EXIT_SUCCESS;
}

int  XXX_PPMD_MatEss2Orient(int argc,char ** argv)
{
   std::vector<double> VMat;
   ElInitArgMain
   (
           argc,argv,
           LArgMain()  << EAMC(VMat,"Matrice as vector of nine double"),
           LArgMain()
   );
   ELISE_ASSERT(VMat.size()==9,"Bad size 4 essential matrix");

   ElMatrix<REAL>  aMat(3,3);
   for (int aX=0 ; aX<3 ; aX++)
   {
      for (int aY=0 ; aY<3 ; aY++)
      {
          aMat(aX,aY) = VMat.at(aX+3*aY);
          // aMat(aY,aX) = VMat.at(aX+3*aY);
      }
   }

   std::list<ElRotation3D> LM=   MatEssToMulipleRot(aMat,1.0);
   for (const auto & aR : LM)
   {
        std::cout << aR.tr() << "\n";
   }


   return EXIT_SUCCESS;
}



/*
int main(int argc,char ** argv)
{
    MMD_InitArgcArgv(argc,argv,2);

    std::string aFilePtsIn,aFilePtsOut;
    bool Help;
    eTypeFichierApp aType;

    std::string aStrType = argv[1];
    StdReadEnum(Help,aType,argv[1],eNbTypeApp);

    std::string aStrChSys;


    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAMC(aStrType,"Type of file")
                      << EAMC(aFilePtsIn,"App File") ,
           LArgMain() << EAM(aFilePtsOut,"Out",true,"Xml Out File")
                      << EAM(aStrChSys,"ChSys",true,"Change coordinate file")
    );



    cChSysCo * aCSC = 0;
    if (aStrChSys!="")
       aCSC = cChSysCo::Alloc(aStrChSys);

    if (aFilePtsOut=="")
    {
        aFilePtsOut =StdPrefixGen(aFilePtsIn) + ".xml";
    }

    ELISE_fp aFIn(aFilePtsIn.c_str(),ELISE_fp::READ);


    cDicoAppuisFlottant  aDico;
    char * aLine;
    int aCpt=0;
    std::vector<Pt3dr> aVPts;
    std::vector<std::string> aVName;
    while ((aLine = aFIn.std_fgets()))
    {
         if (aLine[0] != '#')
         {
            char aName[1000];
            char aTruc[1000];
            double anX,anY,aZ;

            int aNb=0;
            if (aType==eAppEgels)
            {
                aNb = sscanf(aLine,"%s %s %lf %lf %lf",aName,aTruc,&anX,&anY,&aZ);
                if (aNb!=5)
                {
                     std::cout <<  " At line " << aCpt << " of File "<< aFilePtsIn << "\n";
                     ELISE_ASSERT(false,"Could not read the 5 expected values");
                }
            }
            if (aType==eAppGeoCub)
            {
                aNb = sscanf(aLine,"%s %lf %lf %lf",aName,&anX,&anY,&aZ);
                if (aNb!=4)
                {
                     std::cout <<  " At line " << aCpt << " of File "<< aFilePtsIn << "\n";
                     ELISE_ASSERT(false,"Could not read the 4 expected values");
                }
            }
            Pt3dr aP(anX,anY,aZ);
            aVPts.push_back(aP);
            aVName.push_back(aName);
        }
        aCpt ++;
     }


    if (aCSC!=0)
    {
        aVPts = aCSC->Src2Cibl(aVPts);
    }


    for (int aKP=0 ; aKP<int(aVPts.size()) ; aKP++)
    {
        cOneAppuisDAF aOAD;
        aOAD.Pt() = aVPts[aKP];
        aOAD.NamePt() = aVName[aKP];
        aOAD.Incertitude() = Pt3dr(1,1,1);

        aDico.OneAppuisDAF().push_back(aOAD);
    }

    aFIn.close();
    MakeFileXML(aDico,aFilePtsOut);
}

*/





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
