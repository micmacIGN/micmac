#ifndef  _MMVII_DeclareAllCmd_H_
#define  _MMVII_DeclareAllCmd_H_

#include "cMMVII_Appli.h"

namespace MMVII
{


/** \file MMVII_DeclareAllCmd.h
    \brief Contains declaration  of all MMVII Commands
*/

extern cSpecMMVII_Appli  TheSpecBench;
extern cSpecMMVII_Appli  TheSpecTestCpp11;
extern cSpecMMVII_Appli  TheSpecMPDTest;
extern cSpecMMVII_Appli  TheSpecEditSet;
extern cSpecMMVII_Appli  TheSpecEditRel;
extern cSpecMMVII_Appli  TheSpec_EditCalcMetaDataImage;
extern cSpecMMVII_Appli  TheSpecWalkman;
extern cSpecMMVII_Appli  TheSpecDaisy;
extern cSpecMMVII_Appli  TheSpecCatVideo;
extern cSpecMMVII_Appli  TheSpecReduceVideo;
extern cSpecMMVII_Appli  TheSpec_TestEigen;
extern cSpecMMVII_Appli  TheSpec_ComputeParamIndexBinaire;
extern cSpecMMVII_Appli  TheSpecTestRecall;
extern cSpecMMVII_Appli  TheSpecScaleImage;
extern cSpecMMVII_Appli  TheSpec_StackIm;
extern cSpecMMVII_Appli  TheSpecCalcDiscIm;
extern cSpecMMVII_Appli  TheSpecCalcDescPCar;
extern cSpecMMVII_Appli  TheSpecMatchTieP;
extern cSpecMMVII_Appli  TheSpec_TiePConv;
extern cSpecMMVII_Appli  TheSpec_ToTiePMul;
extern cSpecMMVII_Appli  TheSpecEpipGenDenseMatch;
extern cSpecMMVII_Appli  TheSpecEpipDenseMatchEval; 
extern cSpecMMVII_Appli  TheSpecGenSymbDer;
extern cSpecMMVII_Appli  TheSpecKapture;
extern cSpecMMVII_Appli  TheSpecFormatTDEDM_WT;  // Wu Teng
extern cSpecMMVII_Appli  TheSpecFormatTDEDM_MDLB; // Middleburry
extern cSpecMMVII_Appli  TheSpecTestHypStep; // Middleburry
extern cSpecMMVII_Appli  TheSpecExtractLearnVecDM; 
extern cSpecMMVII_Appli  TheSpecCalcHistoCarac; 
extern cSpecMMVII_Appli  TheSpecCalcHistoNDim; 
extern cSpecMMVII_Appli  TheSpecFillCubeCost; 
extern cSpecMMVII_Appli  TheSpecMatchMultipleOrtho; 
extern cSpecMMVII_Appli  TheSpecDMEvalRef; 
extern cSpecMMVII_Appli  TheSpecGenCodedTarget; 
extern cSpecMMVII_Appli  TheSpecExtractCircTarget; 
extern cSpecMMVII_Appli  TheSpecExtractCodedTarget; 
extern cSpecMMVII_Appli  TheSpecSimulCodedTarget; 
extern cSpecMMVII_Appli  TheSpecDensifyRefMatch; 
extern cSpecMMVII_Appli  TheSpecCompletUncodedTarget; 
extern cSpecMMVII_Appli  TheSpecCloudClip; 
extern cSpecMMVII_Appli  TheSpecMeshDev; 
extern cSpecMMVII_Appli  TheSpecGenMeshDev; 
extern cSpecMMVII_Appli  TheSpecTestCovProp; 
extern cSpecMMVII_Appli  TheSpec_OriConvV1V2; 
extern cSpecMMVII_Appli  TheSpec_OriUncalibSpaceResection; 
extern cSpecMMVII_Appli  TheSpec_OriCalibratedSpaceResection; 
extern cSpecMMVII_Appli  TheSpec_OriCheckGCPDist; 
extern cSpecMMVII_Appli  TheSpec_OriRel2Im; 
extern cSpecMMVII_Appli  TheSpecMeshCheck; 
extern cSpecMMVII_Appli  TheSpecProMeshImage; 
extern cSpecMMVII_Appli  TheSpecMeshImageDevlp; 
extern cSpecMMVII_Appli  TheSpecRadiom2ImageSameMod; 
extern cSpecMMVII_Appli  TheSpecRadiomCreateModel;
extern cSpecMMVII_Appli  TheSpecTopoComp;
extern cSpecMMVII_Appli  TheSpecGenerateEncoding;
extern cSpecMMVII_Appli  TheSpecTestGraphPart;
extern cSpecMMVII_Appli  TheSpec_OriBundlAdj;
extern cSpecMMVII_Appli  TheSpecDistCorrectCirgTarget;
extern cSpecMMVII_Appli  TheSpecGenArgsSpec;
extern cSpecMMVII_Appli  TheSpec_ImportGCP;
extern cSpecMMVII_Appli  TheSpec_ImportORGI;
extern cSpecMMVII_Appli  TheSpec_ImportTiePMul;
extern cSpecMMVII_Appli  TheSpec_ImportMesImGCP;
extern cSpecMMVII_Appli  TheSpec_ImportM32;
extern cSpecMMVII_Appli  TheSpec_ConvertV1V2_GCPIM;
extern cSpecMMVII_Appli  TheSpec_SpecSerial;
extern cSpecMMVII_Appli  TheSpec_CGPReport;
extern cSpecMMVII_Appli  TheSpec_TiePReport;
extern cSpecMMVII_Appli  TheSpec_RandomGeneratedDelaunay;
extern cSpecMMVII_Appli  TheSpec_ComputeTriangleDeformation;
extern cSpecMMVII_Appli  TheSpec_ComputeTriangleDeformationTrRad;
extern cSpecMMVII_Appli  TheSpec_ComputeTriangleDeformationTranslation;
extern cSpecMMVII_Appli  TheSpec_ComputeTriangleDeformationRadiometry;
extern cSpecMMVII_Appli  TheSpec_ComputeTriangleDeformationRad;
extern cSpecMMVII_Appli  TheSpec_PoseCmpReport;
extern cSpecMMVII_Appli  TheSpec_BlockCamInit;   // RIGIDBLOC
extern cSpecMMVII_Appli  TheSpec_ClinoInit;
extern cSpecMMVII_Appli  TheSpecRename;
extern cSpecMMVII_Appli  TheSpec_V2ImportCalib;
extern cSpecMMVII_Appli  TheSpec_ImportOri;
extern cSpecMMVII_Appli  TheSpecDicoRename;
extern cSpecMMVII_Appli  TheSpec_SimulDispl;
extern cSpecMMVII_Appli  TheSpec_CreateRTL;
extern cSpecMMVII_Appli  TheSpec_ChSysCo;
extern cSpecMMVII_Appli  TheSpec_ChSysCoGCP;
extern cSpecMMVII_Appli  TheSpec_CreateCalib;
extern cSpecMMVII_Appli  TheSpecImportExtSens;
extern cSpecMMVII_Appli  TheSpecTestSensor;
extern cSpecMMVII_Appli  TheSpecParametrizeSensor;
extern cSpecMMVII_Appli  TheSpec_TutoSerial;
extern cSpecMMVII_Appli  TheSpec_TutoFormalDeriv;
};

#endif  //  _MMVII_DeclareAllCmd_H_
