#include "TestNewOri.h" 

void SaveCSVLine(ofstream & ofs,std::string& Name1,
                 std::string& Name2,std::string& Name3, 
                 double MoyTrDif,double MoyMatDif,double ResIm)
{

   
    ofs << Name1 << "," << Name2 << "," << Name3
        << "," << ToString(MoyTrDif) 
        << "," << ToString(MoyMatDif)
        << "," << ToString(ResIm)
        ;

    ofs << "\n";
}

int TestTriplets_main(int argc,char ** argv)
{

    std::string aPattern;
    std::string aOriDir;
    std::string aCalDir;
    std::string aSH="";
    std::string aOutCSV="TestTrips.csv";
    std::string aOutS="Sims_v2.txt";
    std::string aOutG="GPose_v2.txt";
    std::string aOutE="EGs_v2.txt";

    double dTR_MAX=1;
    double dR_MAX=1;
    double dResIm_MAX=10;

    bool RUN_PAIRS=false;

    cInterfChantierNameManipulateur * aICNM;
    cNewO_NameManager *aNM;


    ElInitArgMain
    (
        argc,argv,
        LArgMain() << EAMC(aPattern,"Pattern of images", eSAM_IsExistFile)
                   << EAMC(aOriDir,"Ori directory", eSAM_IsExistFile),
        LArgMain() << EAM(aSH,"SH",true,"Homol folder postfix, Def=""")
                   << EAM(aCalDir,"Calib",true,"Calibration directory, Def=""")
                   << EAM(aOutS,"OutS",true,"Output file with similarities, Def=Sims_v2.txt")
                   << EAM(aOutG,"OutG",true,"Output file with global poses, Def=GPose_v2.txt")
                   << EAM(aOutE,"OutE",true,"Output file with relative orientations, Def=EGs_v2.txt")
                   << EAM(aOutCSV,"CSV",true,"Output CSV file, Def=TestTrips.csv")
                   << EAM(dTR_MAX,"TrMax",true,"Max allowable error on Tr, Def=1.0")
                   << EAM(dR_MAX,"RMax",true,"Max allowable error on R, Def=1.0")
                   << EAM(dResIm_MAX,"ResImMax",true,"Max allowable cumul reproj error, Def=10.0")
                   << EAM(RUN_PAIRS,"DoPairs",true,"Test also pairs, Def=false")
    );

    //read image pattern 
    aICNM = cInterfChantierNameManipulateur::BasicAlloc("./");
    aNM = new cNewO_NameManager("",aSH,true,"./",aCalDir,"dat");

    StdCorrecNameOrient(aOriDir,"./");

    cElemAppliSetFile anEASF(aPattern);
    const std::vector<std::string> * aSetName =   anEASF.SetIm();

    //open the csv file 
    ofstream aCSVContent;
    if (EAMIsInit(&aOutCSV))
    {
        aCSVContent.open(aOutCSV);
        //aCSVContent<< "Img1,Img2,Img3,dtr1,dMat1,dtr2,dMat2,dtr3,dMat3,dtr,dMat,ResIm";
        aCSVContent<< "Img1,Img2,Img3,dtr,dMat,ResIm";
        aCSVContent<< "\n";
    }

    //open the sim file 
    std::fstream aSimFile;
    if (EAMIsInit(&aOutS))
    {
        aSimFile.open(aOutS.c_str(), std::istream::out);
        aSimFile << std::fixed << setprecision(8) ;
    }

    //open the global pose file 
    std::fstream aGlobFile;
    if (EAMIsInit(&aOutG))
    {
        aGlobFile.open(aOutG.c_str(), std::istream::out);
        aGlobFile << std::fixed << setprecision(8) ;
    }

    //open the edge file 
    std::fstream aEdgeFile;
    if (EAMIsInit(&aOutE))
    {
        aEdgeFile.open(aOutE.c_str(), std::istream::out);
        aEdgeFile << std::fixed << setprecision(8) ;
    }

    //read global poses 
    std::cout << "Read absolute poses " << aSetName->size() << "\n";
    std::map<std::string,ElRotation3D*> aGlobalP; 

    for (auto aImName : (*aSetName))
    {

        std::string aImOriName = aICNM->Dir() + aICNM->Assoc1To1("NKS-Assoc-Im2Orient@-"+aOriDir,aImName,true);
        //std::cout << aImOriName << "\n";

        if (ELISE_fp::exist_file(aImOriName))
        {
            cOrientationConique aOC = StdGetObjFromFile<cOrientationConique>
                                           (
                                               aImOriName,
                                               StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                                               "OrientationConique",
                                               "OrientationConique"
                                           );
            cOrientationExterneRigide anOER = aOC.Externe();

            bool               aTrueR = true;
            ElRotation3D aCRot = Std_RAff_C2M(anOER,anOER.KnownConv().Val());
            Pt3dr tr = aCRot.tr();
            ElMatrix<double> rotation = aCRot.Mat();

            //update
            aGlobalP[aImName] = new ElRotation3D (tr,rotation,aTrueR);

            if (EAMIsInit(&aOutG))
            {
                ElRotation3D *aGlob = aGlobalP[aImName];

                aGlobFile << aImName << " "     << aGlob->Mat()(0,0) << " " << aGlob->Mat()(1,0) << " " << aGlob->Mat()(2,0) << " " 
			    				  << aGlob->Mat()(0,1) << " " << aGlob->Mat()(1,1) << " " << aGlob->Mat()(2,1) << " "
  			    				  << aGlob->Mat()(0,2) << " " << aGlob->Mat()(1,2) << " " << aGlob->Mat()(2,2) << " "
								  << tr.x << " " << tr.y << " " << tr.z << "\n";

                
            }


        }

    }

    //read triplets 
    const ElRotation3D aP1 = ElRotation3D::Id;


    std::string aNameLTriplets = aNM->NameTopoTriplet(true);
    std::string aNameLTripletsNew = StdPrefix(aNameLTriplets)+"_Filt.xml";
    std::string aNameLTripletsNewDmp = StdPrefix(aNameLTriplets)+"_Filt.dmp";

    std::cout << "Read triplets: " << aNameLTriplets  << "\n";
    if (ELISE_fp::exist_file(aNameLTriplets))
    {

        cXml_TopoTriplet  aLT = StdGetFromSI(aNameLTriplets,Xml_TopoTriplet);
        cXml_TopoTriplet  aLTNew;


        int counter=0;
        for (auto a3 : aLT.Triplets())
        {
            //if ( !(counter++ % 10) )
            {
                std::cout << "counter=" << counter << "\n";
            if (DicBoolFind(aGlobalP,a3.Name1()) && DicBoolFind(aGlobalP,a3.Name2()) && DicBoolFind(aGlobalP,a3.Name3()))
            {
                std::string  aName3R = aNM->NameOriOptimTriplet(true,a3.Name1(),a3.Name2(),a3.Name3());
                cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aName3R,Xml_Ori3ImInit);
                double aResIm = aXml3Ori.ResiduTriplet();

                ElRotation3D *aG1 = aGlobalP[a3.Name1()];
                ElRotation3D *aG2 = aGlobalP[a3.Name2()];
                ElRotation3D *aG3 = aGlobalP[a3.Name3()];

                bool aN1InfN2 = (a3.Name1()<a3.Name2());
                bool aN1InfN3 = (a3.Name1()<a3.Name3());
                std::cout << "INF " << aN1InfN2 << " " << a3.Name1() << " " << a3.Name2() << "\n";
                std::cout << "INF " << aN1InfN3 << " " << a3.Name1() << " " << a3.Name3() << "\n";

                ///1-2
                ElRotation3D aP12 = Xml2El(aXml3Ori.Ori2On1());

                ///1-3
                ElRotation3D aP13 = Xml2El(aXml3Ori.Ori3On1());

                ElMatrix<double> Rk(3,3,0.0);
                Pt3dr            Ck;
                double           Lk;
                SimilGlob2LocThreeV(*aG1,*aG2,*aG3,
                                    aP1,aN1InfN2 ? aP12 : aP12.inv(), aN1InfN3 ? aP13 : aP13.inv(),
                                    Rk,Ck,Lk);

                ElRotation3D aG1Pred = ElRotation3D ((Rk.transpose())*(1.0/Lk) *(aP1.tr()-Ck),
                                                     Rk.transpose()*aP1.Mat(),true); 
                ElRotation3D aG2Pred = ElRotation3D ((Rk.transpose())*(1.0/Lk) *(aP12.tr()-Ck),
                                                     Rk.transpose()*aP12.Mat(),true); 
                ElRotation3D aG3Pred = ElRotation3D (Rk.transpose()*(1.0/Lk) *(aP13.tr()-Ck),
                                                     Rk.transpose()*aP13.Mat(),true); 

                //compute errors  
                ElMatrix<double> MatDif1 = aG1->Mat() - aG1Pred.Mat();
                ElMatrix<double> MatDif2 = aG2->Mat() - aG2Pred.Mat();
                ElMatrix<double> MatDif3 = aG3->Mat() - aG3Pred.Mat();

                double MoyTrDif = 0.33*(euclid(aG1->tr()-aG1Pred.tr())+
                                        euclid(aG2->tr()-aG2Pred.tr())+
                                        euclid(aG3->tr()-aG3Pred.tr()));
                double MoyMatDif = 0.33*(sqrt(MatDif1.L2())+
                                         sqrt(MatDif2.L2())+
                                         sqrt(MatDif3.L2()));
 

                //save error to CSV 
                if (EAMIsInit(&aOutCSV))
                {
                    if (dTR_MAX > MoyTrDif)
		            {
                        if (dR_MAX > MoyMatDif)
		                {
                            if (dResIm_MAX > aResIm)
                            {
                                SaveCSVLine(aCSVContent,a3.Name1(),a3.Name2(),a3.Name3(),
                                MoyTrDif,MoyMatDif,aResIm);
                            }
                        }
                    }
                }

                //save similarities 
                if (EAMIsInit(&aOutS))
                {
                    if (dTR_MAX > MoyTrDif)
		    {
                    if (dR_MAX > MoyMatDif)
		    {
                        if (dResIm_MAX > aResIm)
                        {
                            aSimFile << "3 " << a3.Name1() << " " << a3.Name2() << " " << a3.Name3() << " " << 
 	   				        Rk(0,0) << " " << Rk(1,0) << " " << Rk(2,0) << " " << 
                                                Rk(0,1) << " " << Rk(1,1) << " " << Rk(2,1) << " " << 
                                                Rk(0,2) << " " << Rk(1,2) << " " << Rk(2,2) << " " <<
                                                Ck.x << " " << Ck.y << " " << Ck.z << " " << Lk << "\n";
                        }
		    }
		    }
                }

                //save edges 
                if (EAMIsInit(&aOutE))
                {
                    if (dTR_MAX > MoyTrDif)
                    {
                    if (dR_MAX > MoyMatDif)
		    {
                        if (dResIm_MAX > aResIm)
                        {
                            aEdgeFile << "3 " << a3.Name1() << " " << a3.Name2() << " " << " " << a3.Name3() << " "
                                << aP12.Mat()(0,0) << " " << aP12.Mat()(1,0) << " " << aP12.Mat()(2,0) << " "
                                << aP12.Mat()(0,1) << " " << aP12.Mat()(1,1) << " " << aP12.Mat()(2,1) << " "
                                << aP12.Mat()(0,2) << " " << aP12.Mat()(1,2) << " " << aP12.Mat()(2,2) << " "
                                << aP12.tr().x << " "   << aP12.tr().y << " "   << aP12.tr().z   << " "
	 	   			            << aP13.Mat()(0,0) << " " << aP13.Mat()(1,0) << " " << aP13.Mat()(2,0) << " "
                                << aP13.Mat()(0,1) << " " << aP13.Mat()(1,1) << " " << aP13.Mat()(2,1) << " "
                                << aP13.Mat()(0,2) << " " << aP13.Mat()(1,2) << " " << aP13.Mat()(2,2) << " "
                                << aP13.tr().x << " "   << aP13.tr().y << " "   << aP13.tr().z << "\n";
			}
		    }
                    } 
                }

		//save new triplet list 
		if (dTR_MAX > MoyTrDif)
		{
                    if (dR_MAX > MoyMatDif)
		    {
                    if (dResIm_MAX > aResIm)
                    {
		        cXml_OneTriplet aTrip;
		        aTrip.Name1() = a3.Name1();
		        aTrip.Name2() = a3.Name2();
		        aTrip.Name3() = a3.Name3();
                     
	                aLTNew.Triplets().push_back(aTrip);
		    }
		    }
		}
            }

            }
        }
        MakeFileXML(aLTNew,aNameLTripletsNew);
        MakeFileXML(aLTNew,aNameLTripletsNewDmp);
    }


    //read pairs

    if (RUN_PAIRS)
    {
        std::string BogusName="Pair"; 
        std::string aNameLCple = aNM->NameListeCpleOriented(true);
        std::string aNameLCpleNewDmp = StdPrefix(aNameLCple)+"_Filt.dmp";
        std::string aNameLCpleNewXml = StdPrefix(aNameLCple)+"_Filt.xml";
        cSauvegardeNamedRel aLCpl = StdGetFromSI(aNameLCple,SauvegardeNamedRel);
        cSauvegardeNamedRel aLCplNew;
 
        if (ELISE_fp::exist_file(aNameLCple))
        {
            std::cout << "Read pairs: " << aNameLCple  << "\n";
            for (auto a2 : aLCpl.Cple())
            {
                if (DicBoolFind(aGlobalP,a2.N1()) && DicBoolFind(aGlobalP,a2.N2()) )
                {
            	    std::string aN1=a2.N1();
            	    std::string aN2=a2.N2();
                    bool aN1InfN2 = (a2.N1()<a2.N2());
                    std::cout << "aN1InfN2 Pair: " << aN1InfN2 << ", ";

                    cXml_Ori2Im  aXmlO = aNM->GetOri2Im(aN1,aN2);
 
                    double aResIm = aXmlO.Geom().Val().OrientAff().ResiduHighPerc();
                    
            	    bool OK;
            	    ElRotation3D aP2 = aNM->OriCam2On1 (aN1,aN2,OK);
 
                    ElRotation3D *aG1 = aGlobalP[aN1];
                    ElRotation3D *aG2 = aGlobalP[aN2];
 
            	    ElMatrix<double> Rk(3,3,0.0);
            	    Pt3dr            Ck;
            	    double           Lk;
            	    SimilGlob2LocTwoV(*aG1, *aG2,aP1,aN1InfN2 ? aP2.inv() : aP2,Rk,Ck,Lk);
 
            	    //predict global poses
            	    ElRotation3D aG1Pred = ElRotation3D ((Rk.transpose())*(1.0/Lk) *(aP1.inv().tr()-Ck),
                                                         Rk.transpose()*aP1.inv().Mat(),true);
                    ElRotation3D aG2Pred = ElRotation3D ((Rk.transpose())*(1.0/Lk) *(aP2.inv().tr()-Ck),
                                                         Rk.transpose()*aP2.inv().Mat(),true);
 
 
                    //compute errors
                    ElMatrix<double> MatDif1 = aG1->Mat() - aG1Pred.Mat();
                    ElMatrix<double> MatDif2 = aG2->Mat() - aG2Pred.Mat();
 
                    double MoyTrDif = 0.5*(euclid(aG1->tr()-aG1Pred.tr())+
                                            euclid(aG2->tr()-aG2Pred.tr()));
                    double MoyMatDif = 0.5*(sqrt(MatDif1.L2())+
                                             sqrt(MatDif2.L2()));
 
                    std::cout << "aPair: " << euclid(aG1->tr()-aG1Pred.tr()) << " " << euclid(aG2->tr()-aG2Pred.tr()) << "\n";
 
                    //save error to CSV
                    if (EAMIsInit(&aOutCSV))
                    {
                        SaveCSVLine(aCSVContent,BogusName,aN1,aN2,
                                    MoyTrDif,MoyMatDif,aResIm);
                    }
 
            	    //save similarities
                    if (EAMIsInit(&aOutS))
                    {
                        if (dTR_MAX > MoyTrDif)
                        {
                        if (dR_MAX > MoyMatDif)
                        {
                            if (dResIm_MAX > aResIm)
                            {
                                aSimFile << "2 "  << aN1 << " " << aN2 << " " <<
                                                    Rk(0,0) << " " << Rk(1,0) << " " << Rk(2,0) << " " <<
                                                    Rk(0,1) << " " << Rk(1,1) << " " << Rk(2,1) << " " <<
                                                    Rk(0,2) << " " << Rk(1,2) << " " << Rk(2,2) << " " <<
                                                    Ck.x << " " << Ck.y << " " << Ck.z << " " << Lk << "\n";
                            }
                        }
                        }
                    }
 
                    //save edges
                    if (EAMIsInit(&aOutE))
                    {
                        if (dTR_MAX > MoyTrDif)
                        {
                        if (dR_MAX > MoyMatDif)
                        {
                            if (dResIm_MAX > aResIm)
                            {
                                aEdgeFile << "2 " << " " << aN1 << " " << " " << aN2 << " "
                                    << aP2.inv().Mat()(0,0) << " " << aP2.inv().Mat()(1,0) << " " << aP2.inv().Mat()(2,0) << " "
                                    << aP2.inv().Mat()(0,1) << " " << aP2.inv().Mat()(1,1) << " " << aP2.inv().Mat()(2,1) << " "
                                    << aP2.inv().Mat()(0,2) << " " << aP2.inv().Mat()(1,2) << " " << aP2.inv().Mat()(2,2) << " "
                                    << aP2.inv().tr().x << " "     << aP2.inv().tr().y << " "     << aP2.inv().tr().z   << "\n";
                            }
                        }
                        }
                    }
 
            	//save new pair list
                    if (dTR_MAX > MoyTrDif)
                    {
                        if (dR_MAX > MoyMatDif)
                        {
                        if (dResIm_MAX > aResIm)
                        {
            		aLCplNew.Cple().push_back(cCpleString(aN1,aN2));
 
                        }
                        }
                    }
            }
            MakeFileXML(aLCplNew,aNameLCpleNewXml);
            MakeFileXML(aLCplNew,aNameLCpleNewDmp);
        }
        }
    }

    if (EAMIsInit(&aOutCSV))
        aCSVContent.close();
    if (EAMIsInit(&aOutS))
        aSimFile.close();
    if (EAMIsInit(&aOutG))
        aGlobFile.close();
    if (EAMIsInit(&aOutE))
        aEdgeFile.close();


    //translation Ck = ck_i - lambda * Rk * Ci
    //           lam Rk Ci = ck_i - translation 
    //           Ci = 1/L rotation^-1 (ck_i - translation)
    //
    //rotation  Rk = rk_i * Ri^-1
    //          Ri = rotation^-1 rk_i
    //
    //R = alpha.inverse() * r;
    //C = 1.0/lambda * alpha.inverse() * (c - beta);
    //





    return EXIT_SUCCESS;
}
