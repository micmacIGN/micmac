#include "Sat.h"
#include "../uti_phgrm/MICMAC/cCameraModuleOrientation.h"

/** Development based on
CARTOSAT-1 DEM EXTRACTION CAPABILITY STUDY OVER SALON AREA
R. Gachet & P. Fave
**/



class AffCameraASTER
{
public:
	AffCameraASTER(string aFilename, int index, bool activate = false) : filename(StdPrefix(aFilename)), mIndex(index), activated(activate)
	{
		// Loading the GRID file
		ElAffin2D oriIntImaM2C;
		Pt2di Sz(10000, 10000);
		mCamera = new cCameraModuleOrientation(new OrientationGrille(aFilename), Sz, oriIntImaM2C);

		// the 12 parameters of the refine function
		//colc = vP[0] + vP[1] * col + vP[2] * row + vP[6] * sin(2 * M_PI * row / vP[7] + vP[8]);
		//rowc = vP[3] + vP[4] * col + vP[5] * row + vP[9] * sin(2 * M_PI * row / vP[10] + vP[11]);

		//Affinity parameters //cX //cXsX //cXcY //cXcYsY
		// the 6 parameters of affinity
		//vP.push_back(0);// Col : Constant
		//vP.push_back(1);// Col : Param*Col
		//vP.push_back(0);// Col : Param*Row

		//pXx (5th deg poly)
		vP.push_back(0);
		vP.push_back(1);
		vP.push_back(0);
		vP.push_back(0);
		vP.push_back(0);
		vP.push_back(0);
		//pXy (5th deg poly, no 0th order  since already in pXx)
		vP.push_back(0);
		vP.push_back(0);
		vP.push_back(0);
		vP.push_back(0);
		vP.push_back(0);
		//vP.push_back(0);// Row : Constant
		//vP.push_back(0);// Row : Param*Col
		//vP.push_back(1);// Row : Param*Row

		// the 2*3 parameters of jitter 
		//vP.push_back(1);// Col : Amplitude across track of sinusoid
		//vP.push_back(300);// Col : Freq of across track sinusoid
		//vP.push_back(0);// Col : Phase of across track sinusoid

		//Low Freq
		//vP.push_back(0.2);// Row : Amplitude of along track sinusoid (1pix?)
		//vP.push_back(2100);// Row : Freq of along track sinusoid (about 1 cycles per nadir image -> every 5100pix)
		//vP.push_back(0);// Row : Phase of along track sinusoid
		//High Freq
		//vP.push_back(0.2);// Row : Amplitude of along track sinusoid (1pix?)
		//vP.push_back(300);// Row : Freq of along track sinusoid (about 1 cycles per nadir image -> every 5100pix)
		//vP.push_back(0);// Row : Phase of along track sinusoid
	}

	~AffCameraASTER()
	{
		if (mCamera)
			delete mCamera;
	}

	///
	/// \brief update affinity parameters
	/// \param sol unknowns matrix
	///
	void updateParams(ElMatrix <double> const &sol)
	{
		cout << "Init solution of cam " << filename << endl;
		for (size_t aK = 0; aK< vP.size(); aK++)
			cout << vP[aK] << " ";
		cout << endl;

		for (size_t aK = 0; aK < vP.size(); aK++)
		{
			vP[aK] += sol(0, (int)aK);
		}


		cout << "Updated solution: " << endl;
		for (size_t aK = 0; aK< vP.size(); aK++)
			cout << setprecision(15) << vP[aK] << " ";
		cout << endl;
	}

	ElCamera* Camera() { return mCamera; }

	vector <double> vP;

	Pt2dr apply(Pt2dr const &ptImg2)
	{
		/* FOR INFO :
		Pt2dr ptImgC(aA0 + aA1 * ptImg.x + aA2 * ptImg.y + aSX0 * sin(2 * M_PI  * ptImg.x / aSX1 + aSX2),
		aB0 + aB1 * ptImg.x + aB2 * ptImg.y + aSY0 * sin(2 * M_PI  * ptImg.y / aSY1 + aSY2);
		*/

		//affXaffYsY
		//return Pt2dr(vP[0] + vP[1] * ptImg2.x, vP[3] + vP[5] * ptImg2.y + vP[6] * sin(2 * M_PI * ptImg2.y / vP[7] + vP[8]));

		//cXcYsY
		//return Pt2dr(vP[0] + ptImg2.x, vP[1] + ptImg2.y + vP[2] * sin(2 * M_PI * ptImg2.y / vP[3] + vP[4]));

		//cXcY
		//return Pt2dr(vP[0] + ptImg2.x, vP[1] + ptImg2.y);

		//cXsX
		//return Pt2dr(vP[0] + ptImg2.x + vP[1] * sin(2 * M_PI * ptImg2.y / vP[2] + vP[3]), ptImg2.y);

		//pXx
		//return Pt2dr(vP[0] + vP[1] * ptImg2.x + vP[2] * pow(ptImg2.x, 2) + vP[3] * pow(ptImg2.x, 3) + vP[4] * pow(ptImg2.x, 4) + vP[5] * pow(ptImg2.x, 5), ptImg2.y);

		//pXy
		return Pt2dr(vP[0] + vP[1] * ptImg2.x + vP[2] * pow(ptImg2.x, 2) + vP[3] * pow(ptImg2.x, 3) + vP[4] * pow(ptImg2.x, 4) + vP[5] * pow(ptImg2.x, 5) + 
				vP[6] * ptImg2.y + vP[7] * pow(ptImg2.y, 2) + vP[8] * pow(ptImg2.y, 3) + vP[9] * pow(ptImg2.y, 4) + vP[10] * pow(ptImg2.y, 5), ptImg2.y);

		//cX
		//return Pt2dr(vP[0] + ptImg2.x, ptImg2.y);

		//return Pt2dr(ptImg2.x, vP[0] * sin(2 * M_PI * ptImg2.y / vP[1] + vP[2]) + ptImg2.y);
		//return Pt2dr(vP[0] + vP[1] * ptImg2.x + vP[2] * ptImg2.y + vP[6] * sin(2 * M_PI / vP[7] + vP[8]) * ptImg2.x,
		//	vP[3] + vP[4] * ptImg2.x + vP[5] * ptImg2.y + vP[9] * sin(2 * M_PI / vP[10] + vP[11]) * ptImg2.y);
	}

	void activate(bool aVal) { activated = aVal; }
	bool isActivated() { return activated; }

	string name() { return filename; }

	int index() { return mIndex; }

protected:

	///
	/// \brief image filename
	///
	string filename;

	int mIndex;

	ElCamera* mCamera;

	bool activated; //should this camera be estimated

};

class ImageMeasureASTER
{
public:
	ImageMeasureASTER(Pt2dr aPt, AffCameraASTER* aCam) :_ptImg(aPt), _idx(aCam->index()), _imgName(aCam->name()){}

	Pt2dr  pt()       { return _ptImg; }
	string imgName()  { return _imgName; }
	int    imgIndex() { return _idx; }

private:
	Pt2dr _ptImg;       // image coordinates (in pixel)
	int   _idx;         // index of AffCameraASTER

	string _imgName;    // name of AffCameraASTER

	//bool valid; // should this measure be used for estimation
};

class ObservationASTER
{
public:
	ObservationASTER(map <int, AffCameraASTER *> *aMap, string aName = "", bool aValid = false, Pt3dr aPt = Pt3dr(0.f, 0.f, 0.f)) :
		pMapCam(aMap),
		valid(aValid),
		_PtTer(aPt),
		_ptName(aName){}

	virtual Pt3dr getCoord() = 0;

	string ptName() { return _ptName; }

	void addImageMeasureASTER(ImageMeasureASTER const &aMes) { vImgMeasure.push_back(aMes); }

	Pt2dr computeImageDifference(int index,
		Pt3dr pt,
		double apXx0, double apXx1, double apXx2, double apXx3, double apXx4, double apXx5,//pXx
		double apXy1, double apXy2, double apXy3, double apXy4, double apXy5);//pXy
		//double aX0, double aSX0, double aSX1, double aSX2);//cXsX //cX  ;//, double aSY0, double aSY1, double aSY2);

	Pt2dr computeImageDifference(int index, Pt3dr pt);

	map <int, AffCameraASTER*> *pMapCam;

	vector <ImageMeasureASTER> vImgMeasure;

	bool valid; // should this ObservationASTER be used for estimation?

protected:
	Pt3dr _PtTer;

	string _ptName;
};

Pt2dr ObservationASTER::computeImageDifference(int index,
	Pt3dr pt,
	double apXx0, double apXx1, double apXx2, double apXx3, double apXx4, double apXx5,//pXx
	double apXy1, double apXy2, double apXy3, double apXy4, double apXy5)//pXy
	//double aX0, double aSX0, double aSX1, double aSX2) //cXsX//cX //cXcY //cXcYsY , double aSY0, double aSY1, double aSY2)
{
	ImageMeasureASTER* aMes = &vImgMeasure[index];

	Pt2dr ptImg = aMes->pt();

	map<int, AffCameraASTER *>::const_iterator iter = pMapCam->find(aMes->imgIndex());
	AffCameraASTER* cam = iter->second;

	//cXcYsY
	//Pt2dr ptImgC(ptImg.x + aX0, ptImg.y + aY0 + aSY0 * sin(2 * M_PI * ptImg.y / aSY1 + aSY2));

	//cXcY
	//Pt2dr ptImgC(ptImg.x + aX0, ptImg.y + aY0);

	//cX
	//Pt2dr ptImgC(ptImg.x + aX0, ptImg.y);

	//cXsX
	//Pt2dr ptImgC(ptImg.x + aX0 + aSX0 * sin(2 * M_PI * ptImg.y / aSX1 + aSX2), ptImg.y);

	//pXx //pXy
	Pt2dr ptImgC(apXx0 + apXx1 * ptImg.x + apXx2 * pow(ptImg.x, 2) + apXx3 * pow(ptImg.x, 3) + apXx4 * pow(ptImg.x, 4) + apXx5 * pow(ptImg.x, 5) +
			             apXy1 * ptImg.y + apXy2 * pow(ptImg.y, 2) + apXy3 * pow(ptImg.y, 3) + apXy4 * pow(ptImg.y, 4) + apXy5 * pow(ptImg.y, 5), ptImg.y);
	Pt2dr proj = cam->Camera()->R3toF2(pt);
	Pt2dr imageDiff = ptImgC - proj;

	return imageDiff;
}


Pt2dr ObservationASTER::computeImageDifference(int index, Pt3dr pt)
{
	ImageMeasureASTER* aMes = &vImgMeasure[index];

	Pt2dr ptImg = aMes->pt();

	map<int, AffCameraASTER *>::const_iterator iter = pMapCam->find(aMes->imgIndex());
	AffCameraASTER* cam = iter->second;

	Pt2dr ptImgC = cam->apply(ptImg);

	Pt2dr proj = cam->Camera()->R3toF2(pt);

	Pt2dr imageDiff = ptImgC - proj;
	/*
	ofstream fic;
	fic.open("refineASTER/imageDiff.txt", ofstream::app);
	//cout << "Writing refineCoef file : refineASTER/imageDiff.txt" << endl;
	fic << setprecision(15);
	fic << ptImg.x << " " << ptImg.y << " " << ptImgC.x << " " << ptImgC.y << " " << imageDiff.x << " " << imageDiff.y << endl;
	*/
	return imageDiff;
}

class GCPASTER : public ObservationASTER
{
public:
	GCPASTER(map <int, AffCameraASTER*> *aMap, string aPtName = "", bool valid = true, Pt3dr aPt = Pt3dr(0.f, 0.f, 0.f)) :ObservationASTER(aMap, aPtName, valid, aPt){}

	Pt3dr getCoord() { return _PtTer; }
};

class TiePointASTER : public ObservationASTER
{
public:
	TiePointASTER(map <int, AffCameraASTER *> *aMap, string aPtName = "", bool valid = true) : ObservationASTER(aMap, aPtName, valid){}

	Pt3dr getCoord();
};

Pt3dr TiePointASTER::getCoord()
{
	if (vImgMeasure.size() == 2)
	{
		map<int, AffCameraASTER *>::iterator iter1 = pMapCam->find(vImgMeasure[0].imgIndex());
		map<int, AffCameraASTER *>::iterator iter2 = pMapCam->find(vImgMeasure[1].imgIndex());

		AffCameraASTER* cam1 = iter1->second;
		AffCameraASTER* cam2 = iter2->second;

		Pt2dr P1 = cam1->apply(vImgMeasure[0].pt());
		Pt2dr P2 = cam2->apply(vImgMeasure[1].pt());
		return cam1->Camera()->PseudoInter(P1, *cam2->Camera(), P2);
	}
	else
	{
		vector<ElSeg3D>  aVS;
		for (size_t aK = 0; aK < vImgMeasure.size(); aK++)
		{
			map<int, AffCameraASTER *>::iterator iter = pMapCam->find(vImgMeasure[aK].imgIndex());

			if (iter != pMapCam->end())
			{
				AffCameraASTER* aCam = iter->second;

				Pt2dr aPN = aCam->apply(vImgMeasure[aK].pt());

				aVS.push_back(aCam->Camera()->F2toRayonR3(aPN));
			}
		}

		return ElSeg3D::L2InterFaisceaux(0, aVS);
	}
}

double compute2DGroundDifference(Pt2dr const &ptImg1,
	AffCameraASTER* Cam1,
	Pt2dr const &ptImg2,
	AffCameraASTER* Cam2)
{ 
	double z = Cam1->Camera()->PseudoInter(ptImg1, *Cam2->Camera(), ptImg2).z;

	Pt3dr ptTer1 = Cam1->Camera()->ImEtProf2Terrain(ptImg1, z);
	Pt2dr ptImg2C(Cam2->apply(ptImg2));
	Pt3dr ptTer2 = Cam2->Camera()->ImEtProf2Terrain(ptImg2C, z);

	double aGrDif = square_euclid(Pt2dr(ptTer1.x - ptTer2.x, ptTer1.y - ptTer2.y));
	
	/*
	ofstream fic;
	fic.open("refineASTER/imageDiff.txt", ofstream::app);
	//cout << "Writing refineCoef file : refineASTER/imageDiff.txt" << endl;
	fic << setprecision(15);
	fic << ptImg1.x << " " << ptImg1.y << " " << ptImg2.x << " " << ptImg2.y << " " << aGrDif << endl;
	*/

	return aGrDif;
}

//! Abstract class for shared methods
class RefineModelAbsASTER
{
protected:
	map <int, AffCameraASTER*> mapCameras;

	vector <ObservationASTER*> vObs;

	///
	/// \brief normal matrix for least squares estimation
	///
	ElMatrix<double> _N;
	///
	/// \brief matrix for least squares estimation
	///
	ElMatrix<double> _Y;

	size_t numUnk;

	bool _verbose;

	int iteration;

public:

	vector <ObservationASTER*> getObs() { return vObs; }

	///
	/// \brief constructor (loads GRID files, tie-points and filter tie-points on 2D ground difference)
	/// \param aNameFileGridMaster Grid file for master image
	/// \param aNameFileGridSlave Grid file for slave image
	/// \param aNamefileTiePointASTERs Tie-points file
	///
	//cX numUnk(1) //cXsX numUnk(4) //cXcY numUnk(2) //cXcYsY numUnk(5) //pXx numUnk(6) //pXy numUnk(11)
	RefineModelAbsASTER(string const &aFullDir, string imgsExtension = ".tif", bool filter = false) :_N(1, 1, 0.), _Y(1, 1, 0.), numUnk(11), _verbose(false), iteration(0)
	{
		string aDir, aPat;
		SplitDirAndFile(aDir, aPat, aFullDir);

		list<string> aVFiles = RegexListFileMatch(aDir, aPat, 1, false);
		list<string>::iterator itr = aVFiles.begin();
		for (; itr != aVFiles.end(); itr++)
		{
			string aNameFileGrid1 = *itr;

			list<string>::iterator it = itr; it++;
			for (; it != aVFiles.end(); it++)
			{
				string aNameFileGrid2 = *it;

				// Loading the GRID file
				AffCameraASTER* Cam1 = findCamera(aDir + aNameFileGrid1);
				AffCameraASTER* Cam2 = findCamera(aDir + aNameFileGrid2);

				// Loading the Tie Points
				cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

				string aNameFileTiePointASTERs = aDir + aICNM->Assoc1To2
					(
					"NKS-Assoc-CplIm2Hom@@dat",
					StdPrefixGen(aNameFileGrid1) + imgsExtension,
					StdPrefixGen(aNameFileGrid2) + imgsExtension,
					true
					);

				if (!ELISE_fp::exist_file(aNameFileTiePointASTERs)) aNameFileTiePointASTERs = StdPrefixGen(aNameFileTiePointASTERs) + ".txt";

				if (!ELISE_fp::exist_file(aNameFileTiePointASTERs)) cout << "file missing: " << StdPrefixGen(aNameFileTiePointASTERs) + ".dat" << endl;
				else
				{
					cout << "reading: " << aNameFileTiePointASTERs << endl;

					ElPackHomologue aPackHomol = ElPackHomologue::FromFile(aNameFileTiePointASTERs);
					//cout << "taille " << aPackHomol.size() << endl;

					if (aPackHomol.size() == 0)
					{
						cout << "Error in RefineModelAbsASTER: no tie-points" << endl;
						return;
					}

					int rPts_nb = 0; //rejected points number
					int mulTP_nb = 0; //multiple points number

					ElPackHomologue::const_iterator iter = aPackHomol.begin();
					for (; iter != aPackHomol.end(); ++iter)
					{
						ElCplePtsHomologues aCple = iter->ToCple();

						Pt2dr P1 = aCple.P1();
						Pt2dr P2 = aCple.P2();

						//cout << "P1 = "<<P1.x<<" " <<P1.y << endl;
						//cout << "P2 = "<<P2.x<<" " <<P2.y << endl;

						if ((filter) && (compute2DGroundDifference(P1, Cam1, P2, Cam2) > 2500.))//compute2DGroundDifference returns the square of the distance
						{
							rPts_nb++;
							//cout << "Couple with 2D ground difference > 10 rejected" << endl;
						}
						else
						{
							ImageMeasureASTER imMes1(P1, Cam1);
							ImageMeasureASTER imMes2(P2, Cam2);

							bool found = false;

							if (mapCameras.size() > 2)
							{
								//algo brute force (à améliorer)
								for (size_t aK = 0; aK < nObs(); ++aK)
								{
									TiePointASTER* TP = dynamic_cast<TiePointASTER*> (vObs[aK]);
									for (size_t bK = 0; bK < TP->vImgMeasure.size(); bK++)
									{
										ImageMeasureASTER *imMes3 = &TP->vImgMeasure[bK];
										if ((imMes3->imgName() != Cam1->name()) && (imMes3->pt() == P1))
										{
											TP->addImageMeasureASTER(imMes1);
											mulTP_nb++;
											// cout << "multiple point: " << P1.x << " " << P1.y << " found in " << imMes3->idx << " and " << imMes1.idx << endl;
											found = true;
										}
										else if ((imMes3->imgName() != Cam2->name()) && (imMes3->pt() == P2))
										{
											TP->addImageMeasureASTER(imMes2);
											// cout << "multiple point: " << P2.x << " " << P2.y << " found in " << imMes3->idx << " and " << imMes2.idx << endl;
											mulTP_nb++;
											found = true;
										}
									}
								}
							}

							if (!found)
							{
								TiePointASTER *TP = new TiePointASTER(&mapCameras);

								TP->addImageMeasureASTER(imMes1);
								TP->addImageMeasureASTER(imMes2);

								vObs.push_back(TP);

								//cout << "vObs size : " << nObs() << endl;
							}
						}
					}

					cout << "Number of tie points: " << aPackHomol.size() << endl;
					cout << "Number of multiple tie points: " << mulTP_nb << endl;

					if (filter)
					{
						cout << "Number of rejected tie points: " << rPts_nb << endl;
						cout << "Final number of tie points: " << aPackHomol.size() - rPts_nb << endl;
					}
				}
			}
		}

		cout << "mapCameras.size= " << mapCameras.size() << endl;
	}


	///
	/// \brief image distance sum for all tie points (to compute RMS)
	/// \return sum of residuals (square distance - to avoid using sqrt (faster) )
	///
	double sumRes(int &nbMes)
	{
		nbMes = 0;
		double sumRes = 0.;

		//pour chaque point de liaison
		for (size_t aK = 0; aK < nObs(); ++aK)
		{
			ObservationASTER* TP = vObs[aK];
			Pt3dr PT = TP->getCoord();

			//pour chaque image ou le point de liaison est vu
			for (size_t i = 0; i<TP->vImgMeasure.size(); ++i, ++nbMes)
			{
				Pt2dr D = TP->computeImageDifference((int)i, PT);
				//cout << square_euclid(D) << endl;
				sumRes += square_euclid(D);
			}
		}
		return sumRes;
	}

	///
	/// \brief debug matrix
	/// \param mat matrix to write
	///
	void printMatrix(ElMatrix <double> const & mat, string name = "")
	{
		cout << "-------------------------" << endl;
		cout << "Matrix " << name << " : " << endl;
		for (int j = 0; j<mat.Sz().y; ++j)
		{
			for (int i = 0; i<mat.Sz().x; ++i)
				cout << mat(i, j) << " ";

			cout << endl;
		}
		cout << "-------------------------" << endl;
	}

	///
	/// \brief check if a new iteration should be run and write result file (at the step before exiting loop)
	/// \param iniRMS rms before system solve
	/// \param numObs system number of ObservationASTERs
	/// \return
	///
	bool launchNewIter(double iniRMS, int numObs)
	{
		double res = sumRes(numObs);
		cout << "res= " << res << " numObs= " << numObs << endl;


		//ecriture dans un fichier des coefficients en vue d'affiner la grille
		//consommateur en temps => todo: stocker les parametres de l'iteration n-1
		map<int, AffCameraASTER *>::const_iterator iter = mapCameras.begin();

		if (numObs)
		{
			cout << "RMS_init = " << iniRMS << endl;

			double curRMS = sqrt(res / numObs);

			if (curRMS >= iniRMS)
			{
				cout << "curRMS = " << curRMS << " / iniRMS = " << iniRMS << endl;
				cout << "No improve: end at iteration " << iteration << endl;
				return false;
			}
			cout << "RMS_after = " << curRMS << endl;
			iteration++;
			//Print solution to file
			for (size_t aK = 0; iter != mapCameras.end(); ++iter, ++aK)
			{
				AffCameraASTER* cam = iter->second;
				string name = StdPostfix(cam->name());
				ofstream fic(("refineASTER/" + name + "_refineCoef.txt").c_str());
				cout << "Writing refineCoef file : refineASTER" + name + "_refineCoef.txt" << endl;
				fic << setprecision(15);
				//cXsX
				//fic << cam->vP[0] << " 1 0 0 0 1 " << cam->vP[1] << " " << cam->vP[2] << " " << cam->vP[3] << endl;
				//pXx
				//fic << cam->vP[0] << " " << cam->vP[1] <<" 0 0 0 1 " << " " << cam->vP[2] << " " << cam->vP[3] << " " << cam->vP[4] << " " << cam->vP[5] << endl;
				//pXy
				fic << cam->vP[0] << " " << cam->vP[1] << " " << cam->vP[6] << " 0 0 1 " << " " << cam->vP[2] << " " << cam->vP[3] << " " << cam->vP[4] << " " << cam->vP[5] << " " << cam->vP[7] << " " << cam->vP[8] << " " << cam->vP[9] << " " << cam->vP[10] << endl;
				//cX
				//fic << cam->vP[0] << " 1 0 0 0 1" << endl;
				//cXcY
				//fic << cam->vP[0] << " 1 0 " << cam->vP[1] << " 0 1" << endl;
				//cXcYsY
				//fic << cam->vP[0] << " " << cam->vP[1] << " " << cam->vP[2] << " " << cam->vP[4] << " " << cam->vP[5] << endl;
				//fic << cam->vP[0] << " " << cam->vP[1] << " " << cam->vP[2] << " " << cam->vP[3] << " " << cam->vP[4] << " " << cam->vP[5] << " "
				//	<< cam->vP[6] << " " << cam->vP[7] << " " << cam->vP[8] << " " << cam->vP[9] << " " << cam->vP[10] << " " << cam->vP[11] << " " << endl;
			}

			return true;
		}
		else
		{
			cout << "Error in launchNewIter numObs=0" << endl;
			return false;
		}
	}

	///
	/// \brief estimates affinity parameters
	///
	virtual void solve() = 0;

	///
	/// \brief computes the ObservationASTER matrix for one iteration
	/// \return boolean stating if system is solved (need new iteration)
	///
	virtual bool computeObservationASTERMatrix() = 0;

	virtual ~RefineModelAbsASTER(){}

	AffCameraASTER *findCamera(string aFilename)
	{
		map<int, AffCameraASTER *>::const_iterator it = mapCameras.begin();
		for (; it != mapCameras.end(); ++it)
		{
			if (it->second->name() == StdPrefixGen(aFilename)) return it->second;
		}
		AffCameraASTER* Cam = new AffCameraASTER(aFilename, (int)mapCameras.size(), mapCameras.size() != 0);
		mapCameras.insert(pair <int, AffCameraASTER*>((int)mapCameras.size(), Cam));
		return Cam;
	}

	size_t nObs() { return vObs.size(); }
};


//!
class RefineModelGlobalASTER : public RefineModelAbsASTER
{

public:
	RefineModelGlobalASTER(string const &aPattern,
		string const &aImgsExtension,
		string const &aNameFileGCPASTER = "",
		string const &aNameFilePointeIm = "",
		string const &aNameFileMNT = "",
		bool filter = false) :
		RefineModelAbsASTER(aPattern, aImgsExtension, filter)
	{
		if ((aNameFileMNT != "") && (ELISE_fp::exist_file(aNameFileMNT)))
		{
			// Chargement du MNT
			_MntOri = StdGetFromPCP(aNameFileMNT, FileOriMnt);
			if (_verbose) cout << "DTM size : " << _MntOri.NombrePixels().x << " " << _MntOri.NombrePixels().y << endl;
			std_unique_ptr<TIm2D<REAL4, REAL8> > Img(createTIm2DFromFile<REAL4, REAL8>(_MntOri.NameFileMnt()));

#if __cplusplus > 199711L | _MSC_VER == 1800
			_MntImg = std::move(Img);
#else
			_MntImg = Img;
#endif
			if (_MntImg.get() == NULL) cerr << "Error in " << _MntOri.NameFileMnt() << endl;
			//TODO: utiliser le MNT comme contrainte en Z
		}

		if ((aNameFileGCPASTER != "") && (ELISE_fp::exist_file(aNameFileGCPASTER)))
		{
			cDicoAppuisFlottant aDico = StdGetFromPCP(aNameFileGCPASTER, DicoAppuisFlottant);
			if (_verbose)
				cout << "Nb GCPASTER " << aDico.OneAppuisDAF().size() << endl;
			list<cOneAppuisDAF> & aLGCPASTER = aDico.OneAppuisDAF();

			for (
				list<cOneAppuisDAF>::iterator iT = aLGCPASTER.begin();
				iT != aLGCPASTER.end();
			iT++
				)
			{
				if (_verbose) cout << "Point : " << iT->NamePt() << " " << iT->Pt() << endl;

				vObs.push_back(new GCPASTER(&mapCameras, iT->NamePt(), true, iT->Pt()));
			}
		}

		if ((aNameFilePointeIm != "") && (ELISE_fp::exist_file(aNameFilePointeIm)))
		{
			cSetOfMesureAppuisFlottants aDico = StdGetFromPCP(aNameFilePointeIm, SetOfMesureAppuisFlottants);
			if (_verbose)
				cout << "Nb GCPASTER img " << aDico.MesureAppuiFlottant1Im().size() << endl;
			list<cMesureAppuiFlottant1Im> & aLGCPASTER = aDico.MesureAppuiFlottant1Im();

			for (
				list<cMesureAppuiFlottant1Im>::iterator iT = aLGCPASTER.begin();
				iT != aLGCPASTER.end();
			iT++
				)
			{
				if (_verbose)
					cout << "Image : " << StdPrefixGen(iT->NameIm()) << endl;

				list< cOneMesureAF1I > & aLPIm = iT->OneMesureAF1I();

				for (
					list<cOneMesureAF1I>::iterator iTP = aLPIm.begin();
					iTP != aLPIm.end();
				iTP++
					)
				{
					if (_verbose)
						cout << "Point : " << iTP->NamePt() << " " << iTP->PtIm() << endl;

					AffCameraASTER* cam = NULL;
					for (
						map<int, AffCameraASTER *>::iterator iter = mapCameras.begin();
						iter != mapCameras.end();
					iter++
						)
					{
						if (NameWithoutDir(iter->second->name()) == StdPrefixGen(iT->NameIm()))
						{
							cam = iter->second;
							break;
						}
					}

					if (cam)
					{
						ImageMeasureASTER im(iTP->PtIm(), cam);

						for (unsigned int aK = 0; aK<vObs.size(); ++aK)
						{
							if (vObs[aK]->ptName() == iTP->NamePt())
							{
								vObs[aK]->addImageMeasureASTER(im);
								if (_verbose)
									cout << "add img measure to GCPASTER " << vObs[aK]->ptName() << endl;
								break;
							}
						}
					}
				}
			}
		}
	}

	void addObs(int pos, const ElMatrix<double> &obs, const ElMatrix<double> &ccc, const double p, const double res)
	{
		bool verbose = _verbose;

		double pdt = 1. / (p*p);

		ElMatrix <double> C = ccc.transpose()*ccc*pdt;  //1 - ajouter en 0,0
		ElMatrix <double> C1 = ccc.transpose()*obs*pdt;  //2 - 0,(pos-1)*12+3  + sa transposee en (pos-1)*12+3,0
		ElMatrix <double> N1 = obs.transpose()*obs*pdt;  //3 - en (pos-1)*12+3, idem

		//cout << "BANANIA" << endl;
		//cout << C.Sz() << endl;
		//cout << C1.Sz() << endl;
		//cout << N1.Sz() << endl;
		//1 - Ajout de C en (0,0) de _N
		for (int aK = 0; aK < C.Sz().x; aK++)
		for (int bK = 0; bK < C.Sz().y; bK++)
			_N(aK, bK) += C(aK, bK);

		//2 - Ajout de C1 et C1t

		if (verbose) cout << "cam->mIndex : " << pos << endl;

		if (pos > 0)
		{
			if (verbose)  printMatrix(C1, "C1");

			for (int aK = 0; aK < C1.Sz().x; aK++)
			for (int bK = 0; bK < C1.Sz().y; bK++)
				_N((pos - 1)*(int)numUnk + 3 + aK, bK) += C1(aK, bK);

			ElMatrix <double> C1t = C1.transpose();

			if (verbose)  printMatrix(C1t, "C1t");

			for (int aK = 0; aK < C1t.Sz().x; aK++)
			for (int bK = 0; bK < C1t.Sz().y; bK++)
				_N(aK, (pos - 1)*(int)numUnk + 3 + bK) += C1t(aK, bK);

			//3 - Ajout de N1

			if (verbose)  printMatrix(N1, "N1");

			for (int aK = 0; aK < N1.Sz().x; aK++)
			for (int bK = 0; bK < N1.Sz().y; bK++)
				_N((pos - 1)*(int)numUnk + 3 + aK, (pos - 1)*(int)numUnk + 3 + bK) += N1(aK, bK);
		}

		//pour Y
		ElMatrix <double> Y1 = obs.transpose()*res*pdt;
		ElMatrix <double> C2 = ccc.transpose()*res*pdt;

		for (int aK = 0; aK<C2.Sz().y; ++aK)
			_Y(0, aK) += C2(0, aK);

		if (pos > 0)
		for (int aK = 0; aK<Y1.Sz().y; ++aK)
			_Y(0, (pos - 1)*(int)numUnk + 3 + aK) += Y1(0, aK);

		if (verbose)
		{
			printMatrix(_N, "_N");
			printMatrix(_Y, "_Y");
		}
	}

	void addObs(const ElMatrix<double> &ccc, const double p, const double res)
	{
		bool verbose = _verbose;

		double pdt = 1. / (p*p);

		ElMatrix <double> C = ccc.transpose()*ccc*pdt;  //1 - ajouter en 0,0

		//1 - Ajout de C en (0,0) de _N
		for (int aK = 0; aK < C.Sz().x; aK++)
		for (int bK = 0; bK < C.Sz().y; bK++)
			_N(aK, bK) += C(aK, bK);

		//pour Y
		ElMatrix <double> C2 = ccc.transpose()*res*pdt;

		for (int aK = 0; aK<C2.Sz().y; ++aK)
			_Y(0, aK) += C2(0, aK);

		if (verbose)
		{
			printMatrix(_N, "_N");
			printMatrix(_Y, "_Y");
		}
	}

	void addObsStabil(int pos, const ElMatrix<double> &obs, const double p, const double res)
	{
		bool verbose = false;

		double pdt = 1. / (p*p);

		ElMatrix <double> N1 = obs.transpose()*obs*pdt;  //3 - en (pos-1)*6+3, idem

		if (verbose) cout << "cam->mIndex : " << pos << endl;

		if (pos > 0)
		{
			for (int aK = 0; aK < N1.Sz().x; aK++)
			for (int bK = 0; bK < N1.Sz().y; bK++)
				_N((pos - 1)*(int)numUnk + 3 + aK, (pos - 1)*(int)numUnk + 3 + bK) += N1(aK, bK);
		}

		//pour Y
		ElMatrix <double> Y1 = obs.transpose()*res*pdt;

		if (pos > 0)
		for (int aK = 0; aK< Y1.Sz().y; ++aK)
			_Y(0, (pos - 1)*(int)numUnk + 3 + aK) += Y1(0, aK);

		if (verbose)
		{
			printMatrix(_N, "_N");
			printMatrix(_Y, "_Y");
		}
	}

	void solveFirstGroup(vector<int> const &vpos)
	{
		bool verbose = false;

		if (verbose) cout << "solveFirstGroup : " << endl;

		//matrice 3,3 pivot, l'inverser -> c-1 ElSubMat(0,0,3,3)
		ElMatrix <double> C = _N.sub_mat(0, 0, 3, 3);

		if (verbose)   printMatrix(C, "C");

		ElMatrix <double> Cinv = gaussj(C);

		if (verbose)   printMatrix(Cinv, "Cinv");

		//matrice 3,1 pivotY, > Y0 ElSubMat(0,0,3,1)
		ElMatrix <double> Y0 = _Y.sub_mat(0, 0, 1, 3);

		if (verbose)   printMatrix(Y0, "Y0");

		//pour ligne k : extraire ElSubMat(0,k*6+3,6,3) = Dk :
		//si = 0 rien, ligne suivante
		//sinon, extraire de Y sub(0,6*k..,6,1) puis Y -= Dk*C-1*Y0
		//       extraire pour tout col dans vPos sub(ligne k,vPos[k],6,6) puis N -= Dk*C-1*N(0,vPos[k],3,6)

		const int numUnk_int = (int)numUnk;
		for (size_t k = 0; k < vpos.size(); ++k)
		{
			ElMatrix <double> Dk = _N.sub_mat(0, (vpos[k] - 1)*numUnk_int + 3, 3, numUnk_int);

			ElMatrix <double> M2 = Dk*Cinv*Y0;

			if (verbose)   printMatrix(M2, "M2");

			//Y -= Dk*C-1*Y0
			for (int bK = 0; bK < M2.Sz().y; bK++)
				_Y(0, (vpos[k] - 1)*numUnk_int + 3 + bK) -= M2(0, bK);

			if (verbose)
			{
				printMatrix(Dk, "Dk");

				printMatrix(Cinv, "Cinv");
			}

			for (size_t k2 = 0; k2 < vpos.size(); ++k2)
			{
				ElMatrix <double> Dk2 = _N.sub_mat((vpos[k2] - 1)*numUnk_int + 3, 0, numUnk_int, 3);

				//printMatrix(Dk2, "Dk2");

				ElMatrix <double> N2 = Dk*Cinv*Dk2;

				//printMatrix(N2, "N2");

				// N -= Dk*C-1*N(0,vPos[k],3,6)
				for (int aK = 0; aK < N2.Sz().x; aK++)
				for (int bK = 0; bK < N2.Sz().y; bK++)
					_N((vpos[k2] - 1)*numUnk_int + 3 + aK, (vpos[k] - 1)*numUnk_int + 3 + bK) -= N2(aK, bK);
			}
		}

		//RAZ des 1eres lignes et colonnes (3)
		for (int aK = 0; aK<_N.Sz().x; ++aK)
		{
			for (int bK = 0; bK <3; ++bK)
				_N(aK, bK) = 0;
		}

		for (int aK = 0; aK<_N.Sz().y; ++aK)
		{
			for (int bK = 0; bK <3; ++bK)
				_N(bK, aK) = 0;
		}

		for (int aK = 0; aK<3; ++aK)
			_Y(0, aK) = 0;

		if (verbose)
		{
			printMatrix(_N, "_N");
			printMatrix(_Y, "_Y");
		}
	}

	void solve()
	{
		bool verbose = true;

		if (verbose) printMatrix(_N);

		ElMatrix<double> Nsub = _N.sub_mat(3, 3, _N.Sz().x - 3, _N.Sz().y - 3);
		ElMatrix<double> Ysub = _Y.sub_mat(0, 3, 1, _Y.Sz().y - 3);

		ElMatrix<double> inv = gaussj(Nsub);
		if (verbose)  printMatrix(inv, "inv");

		ElMatrix<double> sol = inv*Ysub;

		if (verbose) printMatrix(sol, "sol");

		//cout << "SOL_NORM = " << sol.NormC(2) << endl;



		//for (size_t aK=0; aK < vObs.size(); aK++)
		for (size_t aK = 0; aK < 10; aK++)
		{
			ObservationASTER* aObs = vObs[aK];
			cout << aObs->getCoord().z << endl;
		}


		int aK = 0;
		map<int, AffCameraASTER *>::const_iterator iter = mapCameras.begin();
		iter++; //don't use first image
		const int numUnk_int = (int)numUnk;
		for (; iter != mapCameras.end(); ++iter)
		{
			ElMatrix <double> solSub = sol.sub_mat(0, aK, 1, numUnk_int);
			iter->second->updateParams(solSub);

			aK += numUnk_int;
		}


		for (size_t aK = 0; aK < 10; aK++)
		{
			ObservationASTER* aObs = vObs[aK];
			cout << aObs->getCoord().z << endl;
		}
	}

	//! compute the ObservationASTER matrix for one iteration
	bool computeObservationASTERMatrix()
	{
		cout << "iter=" << iteration << endl;

		double NoData = -9999.;

		int numObs;
		double res = sumRes(numObs);
		cout << "res= " << res << " numObs= " << numObs << endl;
		if (numObs)
		{
			double iniRMS = sqrt(res / numObs);
			//cout << "RMS_ini = " << iniRMS << endl;
			/* FOR INFO :
			Pt2dr ptImgC(aX0 + aX1 * ptImg.x + aX2 * ptImg.y + aSX0 * sin(2 * M_PI  * ptImg.x / aSX1 + aSX2),
			aY0 + aY1 * ptImg.x + aY2 * ptImg.y + aSY0 * sin(2 * M_PI  * ptImg.y / aSY1 + aSY2);
			*/

			double dX = 0.1;
			double dY = 0.1;
			double dZ = 0.1;

			//pXx //cXsX //cX //cXcY //cXcYsY
			double dX0 = 0.1;
			double dX1 = 0.1;
			double dX2 = 0.01;
			double dX3 = 0.001;
			double dX4 = 0.001;
			double dX5 = 0.001;
			//pXy
			double dXy1 = 0.1;
			double dXy2 = 0.01;
			double dXy3 = 0.001;
			double dXy4 = 0.001;
			double dXy5 = 0.001;
			//double dSX0 = 0.05;
			//double dSX1 = 1;
			//double dSX2 = 0.05;
			//double dY0 = 0.05;
			//double dY1 = 0.1;
			//double dY2 = 0.1;
			//double dSY0 = 0.05;
			//double dSY1 = 1;
			//double dSY2 = 0.05;

			//Get number of cameras to estimate
			int nbCam = 0;
			map<int, AffCameraASTER *>::const_iterator it = mapCameras.begin();
			for (; it != mapCameras.end(); ++it)
			if (it->second->isActivated()) nbCam++;

			cout << "Nb cam to estimate : " << nbCam << endl;

			//Init matrix
			const int numUnk_int = (int)numUnk;
			int matSz = 3 + numUnk_int*nbCam;
			cout << "matSz : " << matSz << endl;
			cout << "numUnk : " << numUnk << endl;
			_N = ElMatrix<double>(matSz, matSz);
			_Y = ElMatrix<double>(1, matSz);

			//pour chaque ObservationASTER
			for (size_t aK = 0; aK < nObs(); aK++)
			{
				ObservationASTER* aObs = vObs[aK];

				vector <ImageMeasureASTER> vMes = aObs->vImgMeasure;
				vector <int> vPos;

				Pt3dr pt = aObs->getCoord();

				//pour chaque image où le point de liaison est vu
				for (size_t bK = 0; bK < vMes.size(); ++bK)
				{
					Pt2dr D = aObs->computeImageDifference((int)bK, pt);
					// double ecart2 = square_euclid(D);

					double pdt = 1.; //1./sqrt(1. + ecart2);

					//todo : strategie d'elimination d'ObservationASTERs / ou ponderation

					ElMatrix<double> obs(numUnk_int, 1);
					ElMatrix<double> ccc(3, 1);

					// estimation des derivees partielles
					map<int, AffCameraASTER *>::iterator iter = mapCameras.find(vMes[bK].imgIndex());
					AffCameraASTER* cam = iter->second;

					if (cam->index() > 0) vPos.push_back(cam->index());

					//For every Parameters
					//pXx 
					double X0 = cam->vP[0];
					double X1 = cam->vP[1];
					double X2 = cam->vP[2];
					double X3 = cam->vP[3];
					double X4 = cam->vP[4];
					double X5 = cam->vP[5];
					//pXy 
					double Xy1 = cam->vP[6];
					double Xy2 = cam->vP[7];
					double Xy3 = cam->vP[8];
					double Xy4 = cam->vP[9];
					double Xy5 = cam->vP[10];
					//cX 
					//double X0 = cam->vP[0];
					//double X1 = cam->vP[1];
					//double X2 = cam->vP[2];
					//double Y0 = cam->vP[3];
					//double Y1 = cam->vP[4];
					//double Y2 = cam->vP[5];
					//double SY0 = cam->vP[6];
					//double SY1 = cam->vP[7];
					//double SY2 = cam->vP[8];

					//cXcYsY //cXcY
					//double X0 = cam->vP[0];
					//double Y0 = cam->vP[1];
					//double SY0 = cam->vP[2];
					//double SY1 = cam->vP[3];
					//double SY2 = cam->vP[4];

					//cXsX
					//double X0 = cam->vP[0];
					//double SX0 = cam->vP[1];
					//double SX1 = cam->vP[2];
					//double SX2 = cam->vP[3];

					/* FOR INFO :
					Pt2dr ptImgC(aX0 + aX1 * ptImg.x + aX2 * ptImg.y + aSX0 * sin(2 * M_PI  * ptImg.x / aSX1 + aSX2),
					aY0 + aY1 * ptImg.x + aY2 * ptImg.y + aSY0 * sin(2 * M_PI  * ptImg.y / aSY1 + aSY2);
					*/

					//Sinus in y
					//cX 
					//Pt2dr vdX0 = Pt2dr(1. / dX0, 1. / dX0) * (aObs->computeImageDifference(bK, pt, X0 + dX0) - D);// , SY0, SY1, SY2) - D);
					//Pt2dr vdX = Pt2dr(1. / dX, 1. / dX) * (aObs->computeImageDifference(bK, Pt3dr(pt.x + dX, pt.y, pt.z), X0) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdY = Pt2dr(1. / dY, 1. / dY) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y + dY, pt.z), X0) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdZ = Pt2dr(1. / dZ, 1. / dZ) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y, pt.z + dZ), X0) - D);//, SY0, SY1, SY2) - D);
					//pXx
					//Pt2dr vdX0 = Pt2dr(1. / dX0, 1. / dX0) * (aObs->computeImageDifference(bK, pt, X0 + dX0, X1, X2, X3, X4, X5) - D);
					//Pt2dr vdX1 = Pt2dr(1. / dX1, 1. / dX1) * (aObs->computeImageDifference(bK, pt, X0, X1 + dX1, X2, X3, X4, X5) - D);
					//Pt2dr vdX2 = Pt2dr(1. / dX2, 1. / dX2) * (aObs->computeImageDifference(bK, pt, X0, X1, X2 + dX2, X3, X4, X5) - D);
					//Pt2dr vdX3 = Pt2dr(1. / dX3, 1. / dX3) * (aObs->computeImageDifference(bK, pt, X0, X1, X2, X3 + dX3, X4, X5) - D);
					//Pt2dr vdX4 = Pt2dr(1. / dX4, 1. / dX4) * (aObs->computeImageDifference(bK, pt, X0, X1, X2, X3, X4 + dX4, X5) - D);
					//Pt2dr vdX5 = Pt2dr(1. / dX5, 1. / dX5) * (aObs->computeImageDifference(bK, pt, X0, X1, X2, X3, X4, X5 + dX5) - D);
					//Pt2dr vdX = Pt2dr(1. / dX, 1. / dX) * (aObs->computeImageDifference(bK, Pt3dr(pt.x + dX, pt.y, pt.z), X0, X1, X2, X3, X4, X5) - D);
					//Pt2dr vdY = Pt2dr(1. / dY, 1. / dY) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y + dY, pt.z), X0, X1, X2, X3, X4, X5) - D);
					//Pt2dr vdZ = Pt2dr(1. / dZ, 1. / dZ) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y, pt.z + dZ), X0, X1, X2, X3, X4, X5) - D);
					//pXy
					Pt2dr vdX0 = Pt2dr(1. / dX0, 1. / dX0) * (aObs->computeImageDifference((int)bK, pt, X0 + dX0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdX1 = Pt2dr(1. / dX1, 1. / dX1) * (aObs->computeImageDifference((int)bK, pt, X0, X1 + dX1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdX2 = Pt2dr(1. / dX2, 1. / dX2) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2 + dX2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdX3 = Pt2dr(1. / dX3, 1. / dX3) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3 + dX3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdX4 = Pt2dr(1. / dX4, 1. / dX4) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4 + dX4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdX5 = Pt2dr(1. / dX5, 1. / dX5) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5 + dX5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdXy1 = Pt2dr(1. / dX1, 1. / dX1) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5, Xy1 + dXy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdXy2 = Pt2dr(1. / dX2, 1. / dX2) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5, Xy1, Xy2 + dXy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdXy3 = Pt2dr(1. / dX3, 1. / dX3) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3 + dXy3, Xy4, Xy5) - D);
					Pt2dr vdXy4 = Pt2dr(1. / dX4, 1. / dX4) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4 + dXy4, Xy5) - D);
					Pt2dr vdXy5 = Pt2dr(1. / dX5, 1. / dX5) * (aObs->computeImageDifference((int)bK, pt, X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5 + dXy5) - D);
					Pt2dr vdX = Pt2dr(1. / dX, 1. / dX) * (aObs->computeImageDifference((int)bK, Pt3dr(pt.x + dX, pt.y, pt.z), X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdY = Pt2dr(1. / dY, 1. / dY) * (aObs->computeImageDifference((int)bK, Pt3dr(pt.x, pt.y + dY, pt.z), X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					Pt2dr vdZ = Pt2dr(1. / dZ, 1. / dZ) * (aObs->computeImageDifference((int)bK, Pt3dr(pt.x, pt.y, pt.z + dZ), X0, X1, X2, X3, X4, X5, Xy1, Xy2, Xy3, Xy4, Xy5) - D);
					//cXsX
					//Pt2dr vdX0 = Pt2dr(1. / dX0, 1. / dX0) * (aObs->computeImageDifference(bK, pt, X0 + dX0, SX0, SX1, SX2) - D);// , SY0, SY1, SY2) - D);
					//Pt2dr vdX = Pt2dr(1. / dX, 1. / dX) * (aObs->computeImageDifference(bK, Pt3dr(pt.x + dX, pt.y, pt.z), X0, SX0, SX1, SX2) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdY = Pt2dr(1. / dY, 1. / dY) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y + dY, pt.z), X0, SX0, SX1, SX2) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdZ = Pt2dr(1. / dZ, 1. / dZ) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y, pt.z + dZ), X0, SX0, SX1, SX2) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdSX0 = Pt2dr(1. / dSX0, 1. / dSX0) * (aObs->computeImageDifference(bK, pt, X0, SX0 + dSX0, SX1, SX2) - D);
					//Pt2dr vdSX1 = Pt2dr(1. / dSX1, 1. / dSX1) * (aObs->computeImageDifference(bK, pt, X0, SX0, SX1 + dSX1, SX2) - D);
					//Pt2dr vdSX2 = Pt2dr(1. / dSX2, 1. / dSX2) * (aObs->computeImageDifference(bK, pt, X0, SX0, SX1, SX2 + dSX2) - D);
					//cXcY 
					//Pt2dr vdX0 = Pt2dr(1. / dX0, 1. / dX0) * (aObs->computeImageDifference(bK, pt, X0 + dX0, Y0) - D);// , SY0, SY1, SY2) - D);
					//Pt2dr vdY0 = Pt2dr(1. / dY0, 1. / dY0) * (aObs->computeImageDifference(bK, pt, X0, Y0 + dY0) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdX = Pt2dr(1. / dX, 1. / dX) * (aObs->computeImageDifference(bK, Pt3dr(pt.x + dX, pt.y, pt.z), X0, Y0) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdY = Pt2dr(1. / dY, 1. / dY) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y + dY, pt.z), X0, Y0) - D);//, SY0, SY1, SY2) - D);
					//Pt2dr vdZ = Pt2dr(1. / dZ, 1. / dZ) * (aObs->computeImageDifference(bK, Pt3dr(pt.x, pt.y, pt.z + dZ), X0, Y0) - D);//, SY0, SY1, SY2) - D);
					//cXcYsY
					//Pt2dr vdSY0 = Pt2dr(1. / dSY0, 1. / dSY0) * (aObs->computeImageDifference(bK, pt, X0, Y0, SY0 + dSY0, SY1, SY2) - D);
					//Pt2dr vdSY1 = Pt2dr(1. / dSY1, 1. / dSY1) * (aObs->computeImageDifference(bK, pt, X0, Y0, SY0, SY1 + dSY1, SY2) - D);
					//Pt2dr vdSY2 = Pt2dr(1. / dSY2, 1. / dSY2) * (aObs->computeImageDifference(bK, pt, X0, Y0, SY0, SY1, SY2 + dSY2) - D);

					if (cam->index() != 0)
					{

						//Shift in Col and Row //cX
						//obs(0, 0) = vdX0.x;

						//Poly Xx //pXx
						obs(0, 0) = vdX0.x;
						obs(1, 0) = vdX1.x;
						obs(2, 0) = vdX2.x;
						obs(3, 0) = vdX3.x;
						obs(4, 0) = vdX4.x;
						obs(5, 0) = vdX5.x;
						//Poly Xy //pXy
						obs(6, 0) = vdXy1.x;
						obs(7, 0) = vdXy2.x;
						obs(8, 0) = vdXy3.x;
						obs(9, 0) = vdXy4.x;
						obs(10, 0) = vdXy5.x;
						//Shift in Col and Row //cXsX
						//obs(0, 0) = vdX0.x;
						//obs(1, 0) = vdSX0.x;
						//obs(2, 0) = vdSX1.x;
						//obs(3, 0) = vdSX2.x;

						//Shift in Col and Row //cXcY
						//obs(0, 0) = vdX0.x;
						//obs(1, 0) = vdY0.x;

						//Sin in Row //cXcYsY
						//obs(2, 0) = vdSY0.x;
						//obs(3, 0) = vdSY1.x;
						//obs(4, 0) = vdSY2.x;

					}

					ccc(0, 0) = vdX.x;
					ccc(1, 0) = vdY.x;
					ccc(2, 0) = vdZ.x;

					addObs(cam->index(), obs, ccc, pdt, -D.x);

					if (cam->index() != 0)
					{

						//Shift in Col and Row //cX
						//obs(0, 0) = vdX0.y;

						//Poly Xx //pXx
						obs(0, 0) = vdX0.y;
						obs(1, 0) = vdX1.y;
						obs(2, 0) = vdX2.y;
						obs(3, 0) = vdX3.y;
						obs(4, 0) = vdX4.y;
						obs(5, 0) = vdX5.y;
						//Poly Xy //pXy
						obs(6, 0) = vdXy1.y;
						obs(7, 0) = vdXy2.y;
						obs(8, 0) = vdXy3.y;
						obs(9, 0) = vdXy4.y;
						obs(10, 0) = vdXy5.y;

						//Shift in Col and Row //cXsX
						//obs(0, 0) = vdX0.y;
						//obs(1, 0) = vdSX0.y;
						//obs(2, 0) = vdSX1.y;
						//obs(3, 0) = vdSX2.y;

						//Shift in Col and Row //cXcY
						//obs(0, 0) = vdX0.y;
						//obs(1, 0) = vdY0.y;

						//Sin in Row //cXcYsY
						//obs(2, 0) = vdSY0.y;
						//obs(3, 0) = vdSY1.y;
						//obs(4, 0) = vdSY2.y;
					}

					ccc(0, 0) = vdX.y;
					ccc(1, 0) = vdY.y;
					ccc(2, 0) = vdZ.y;

					addObs(cam->index(), obs, ccc, pdt, -D.y);
				}

				solveFirstGroup(vPos);

				//Contrainte en Z
				if (_MntImg.get() != NULL)
				{
					Pt2dr ptMnt(pt.x, pt.y);
					double zMnt = _MntImg->getr(ptMnt, NoData)*_MntOri.ResolutionAlti() + _MntOri.OrigineAlti();

					if (zMnt == NoData)
						cout << "No altitude found for the point : " << pt.x << " " << pt.y << endl;
					else
					{
						double res = -pt.z + zMnt;

						ElMatrix<double> ccd(3, 1);
						ccd(0, 0) = 0;
						ccd(1, 0) = 0;
						ccd(2, 0) = 1;

						double sigma = 20.;

						addObs(ccd, sigma, res);
					}
				}
			}

			map<int, AffCameraASTER *>::iterator iter = mapCameras.begin();
			for (; iter != mapCameras.end(); ++iter)
			{
				AffCameraASTER* cam = iter->second;

				for (size_t aK = 0; aK < numUnk; aK++)
				{
					ElMatrix <double> AB((int)numUnk, 1);
					AB((int)aK, 0) = 1.;

					double sig = 1.;
					if ((aK == 0) || (aK == 3)) sig = 0.1;  //pix
					else sig = 1e-4;

					/*if ((aK==1) || (aK==5))
					addObsStabil(cam->mIndex, AB, sig, 1. - cam->vP[aK]);*/
					/*else
					addObsStabil(cam->mIndex, AB, sig, 0. - cam->vP[aK]);*/

					if ((aK == 0) || (aK == 3))
						addObsStabil(cam->index(), AB, sig, 0. - cam->vP[aK]);
				}
			}

			cout << "before solve" << endl;

			solve();

			return launchNewIter(iniRMS, numObs);
		}
		else
		{
			cout << "Error in computeObservationASTERMatrix numObs=0" << endl;
			return false;
		}
	}

	~RefineModelGlobalASTER()
	{
	}

private:
	std_unique_ptr<TIm2D<REAL4, REAL8> > _MntImg;
	cFileOriMnt                        _MntOri;
};

int RefineJitter_main(int argc, char **argv)
{
	string aPat; // GRID files pattern
	string aImgsExtension = ".tif"; //img file extension (such used in Tapioca)
	string aNameMNT = ""; //DTM file
	string aNamePointeIm = ""; //Pointe image file
	string aNameGCPASTER = ""; //GCPASTER file
	bool filterInput = false;
	bool exportResidus = false;

	ElInitArgMain
		(
		argc, argv,
		LArgMain() << EAMC(aPat, "GRID files pattern")
		<< EAMC(aImgsExtension, "Img files extension"),
		LArgMain() << EAM(aNameGCPASTER, "GCPASTER", true, "GCPASTER file")
		<< EAM(aNamePointeIm, "IMG", true, "Pointe image file")
		<< EAM(aNameMNT, "DTM", true, "DTM file")
		<< EAM(filterInput, "Filter", true, "Remove TiePointASTERs with ground distance > 10m (def=false)")
		<< EAM(exportResidus, "ExpRes", true, "Export residuals (def=false)")
		);

	ELISE_fp::MkDirSvp("refineASTER");

	RefineModelGlobalASTER model(aPat, aImgsExtension, aNameGCPASTER, aNamePointeIm, aNameMNT, filterInput);

	bool ok = (model.nObs() > 3);
	for (size_t iter = 0; (iter < 100) & ok; iter++)
		ok = model.computeObservationASTERMatrix();

	if (exportResidus)
	{
		ofstream ficRes("refine/residus.txt");
		ofstream ficGlb("refine/residusGlob.txt");
		ficRes << setprecision(15);
		ficGlb << setprecision(15);

		vector <ObservationASTER*> vTP = model.getObs();
		for (size_t aK = 0; aK < vTP.size(); ++aK)
		{
			ObservationASTER* aObs = vTP[aK];

			Pt3dr PT = aObs->getCoord();

			double sumRes = 0.;
			for (size_t i = 0; i<aObs->vImgMeasure.size(); ++i)
			{
				Pt2dr D = aObs->computeImageDifference((int)i, PT);

				ficRes << aK << " " << aObs->vImgMeasure[i].imgName() << " " << aObs->vImgMeasure[i].pt().x << " " << aObs->vImgMeasure[i].pt().y << " " << D.x << " " << D.y << " " << endl;

				sumRes += square_euclid(D);
			}

			ficGlb << aK << " " << PT.x << " " << PT.y << " " << PT.z << " " << sumRes << " " << sqrt(sumRes / aObs->vImgMeasure.size()) << endl;
		}
	}

	return EXIT_SUCCESS;
}
