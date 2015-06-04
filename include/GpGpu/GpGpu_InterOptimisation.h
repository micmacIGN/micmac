#ifndef __OPTIMISATION_H__
#define __OPTIMISATION_H__

#include "GpGpu/SData2Optimize.h"
#include "GpGpu/GpGpu_MultiThreadingCpu.h"
#include "GpGpu/GpGpu_eLiSe.h"

extern "C" void Gpu_OptimisationOneDirection(DEVC_Data2Opti  &d2O);

template <class T>
///
/// \brief The CuHostDaPo3D struct
/// Structure 1D des couts de corélation
struct sMatrixCellCost
{
	///
	/// \brief _CostInit1D
	/// les couts intrinséques des cellules
	CuHostData3D<T>         _CostInit1D;
	///
	/// \brief _ptZ
	/// Coordonnées Z de la nappe d'une position terrain
	CuHostData3D<short3>    _ptZ;
	///
	/// \brief _dZ
	/// Delta Z
    CuHostData3D<ushort>    _dZ;
	///
	/// \brief _pit
	/// Decallage pour acceder au début d'une  ligne
    CuHostData3D<uint>      _pit;

	///
	/// \brief _size
	/// Taille totale de _CostInit1D
    uint                    _size;

	///
	/// \brief _maxDz
	/// Delta z Maximum
    ushort                  _maxDz;

    sMatrixCellCost():
        _maxDz(NAPPEMAX) // ATTENTION : NAPPE Dynamique
	{
		_CostInit1D.setAlignMemory(false);

	}

	///
	/// \brief ReallocPt Reallocation de la mémoire conditionelle
	/// \param dim
	///
    void                    ReallocPt(uint2 dim)
    {
        _ptZ.ReallocIf(dim);
        _dZ.ReallocIf(dim);
        _pit.ReallocIf(dim);
        _size = 0;
    }

	///
	/// \brief ReallocData Reallocation de la mémoire conditionelle
	///
    void                    ReallocData()
    {
        _CostInit1D.ReallocIf(_size);

    }

	///
	/// \brief fillCostInit
	/// \param val
	/// Remplir la strucutre par la donnée val
    void                    fillCostInit(ushort val)
    {
        _CostInit1D.Fill(val);
    }

	///
	/// \brief Dealloc Désalocation de la mémoire
	///
    void                    Dealloc()
    {
        _CostInit1D.Dealloc();

        _ptZ.Dealloc();
        _dZ.Dealloc();
        _pit.Dealloc();
	}

	///
	/// \brief PointIncre
	/// \param pt
	/// \param ptZ
	///
    void PointIncre(uint2 pt,short2 ptZ)
    {
		ushort maxNappe = 2048;

        ushort dZ   = abs(count(ptZ));
        _ptZ[pt]    = make_short3(ptZ.x,ptZ.y,0);
		_dZ[pt]     = dZ;

		if(dZ > maxNappe)
		{
			_dZ[pt] = maxNappe;
			dZ		= maxNappe;
			_ptZ[pt] = make_short3(ptZ.x,ptZ.x + maxNappe,0);
		}

        // NAPPEMAX
        if(_maxDz < dZ) // Calcul de la taille de la Nappe Max pour le calcul Gpu
			_maxDz = sgpu::__multipleSup<WARPSIZE>(dZ);

        _pit[pt]    = _size;
		if(_CostInit1D.alignMemory())
		{
			//int adZ = iDivUp(dZ,4) * 4;
			int adZ = sgpu::__multipleSup<4>(dZ);
			_size      += adZ;
		}
		else
			_size      += dZ;
    }

	///
	/// \brief setDefCor Définir la valeur par défaut de corrélation
	/// \param pt
	/// \param defCor
	///
    void                    setDefCor(uint2 pt,short defCor)
    {
        _ptZ[pt].z    = defCor;
    }

	///
	/// \brief Pit
	/// \param pt
	/// \return le décalage pour un point pt
	///
    uint                    Pit(uint2 pt)
    {
        return _pit[pt];
    }

	///
	/// \brief Pit
	/// \param pt
	/// \return  le décalage pour un point pt
	///
    uint                    Pit(Pt2di pt)
    {
        return _pit[toUi2(pt)];
    }

	///
	/// \brief PtZ
	/// \param pt
	/// \return Les coordonnées Zmin et Zmax
	///
    short3                  PtZ(uint2 pt)
    {
        return _ptZ[pt];
    }

	///
	/// \brief PtZ
	/// \param pt
	/// \return  Les coordonnées Zmin et Zmax
	///
    short3                  PtZ(Pt2di pt)
    {
        return _ptZ[toUi2(pt)];
    }

	///
	/// \brief DZ
	/// \param pt
	/// \return le delta Z
	///
    ushort                  DZ(uint2 pt)
    {
        return _dZ[pt];
    }

	///
	/// \brief DZ
	/// \param pt
	/// \return  le delta Z
	///
    ushort                  DZ(Pt2di pt)
    {
        return _dZ[toUi2(pt)];
    }

	///
	/// \brief DZ
	/// \param ptX
	/// \param ptY
	/// \return le delta Z
	///
    ushort                  DZ(uint ptX,uint ptY)
    {
        return _dZ[make_uint2(ptX,ptY)];
    }

	///
	/// \brief Size
	/// \return La taille de la structure
	///
    uint                    Size()
    {
        return _size;
    }       

	///
	/// \brief operator []
	/// \param pt
	/// \return le cout intrinsèque du point
	///
    T*                      operator[](uint2 pt)
    {
        return _CostInit1D.pData() + _pit[pt];
    }

	///
	/// \brief operator []
	/// \param pt
	/// \return  le cout intrinsèque du point
	///
    T*                      operator[](Pt2di pt)
    {
        return _CostInit1D.pData() + _pit[toUi2(pt)];
    }

	///
	/// \brief operator []
	/// \param pt
	/// \return   le cout intrinsèque du point
	///
    T&                      operator[](int3 pt)
    {
        uint2 ptTer = make_uint2(pt.x,pt.y);
        return *(_CostInit1D.pData() + _pit[ptTer] - _ptZ[ptTer].x + pt.z);
    }
};


/// \class InterfOptimizGpGpu
/// \brief Class qui permet a micmac de lancer les calculs d optimisations sur le Gpu
///
class InterfOptimizGpGpu : public CSimpleJobCpuGpu<bool>
{

public:
    ///
    /// \brief InterfOptimizGpGpu
    ///
    InterfOptimizGpGpu();
    ~InterfOptimizGpGpu();

    ///
	/// \brief Data2Opt
	/// \return les données device
    ///
    HOST_Data2Opti& HData2Opt(){ return _H_data2Opt;}
	///
	/// \brief DData2Opt
	/// \return Les données hote
	///
    DEVC_Data2Opti& DData2Opt(){ return _D_data2Opt;}

    ///
    /// \brief Dealloc
    ///
    void            Dealloc();

	///
	/// \brief Prepare Initialisation des paramètres
	/// \param x
	/// \param y
	/// \param penteMax
	/// \param NBDir
	/// \param zReg
	/// \param zRegQuad
	/// \param costDefMask
	/// \param costDefMaskTrans
	/// \param hasMaskAuto
	///
    void            Prepare(uint x, uint y, ushort penteMax, ushort NBDir, float zReg, float zRegQuad, ushort costDefMask, ushort costDefMaskTrans, bool hasMaskAuto);

    ///
    /// \brief freezeCompute
    ///
    void            freezeCompute();

	///
	/// \brief _preFinalCost1D
	/// Structure de données des couts forcées
    CuHostData3D<uint>      _preFinalCost1D;

	///
	/// \brief _FinalDefCor
	/// Structure de données des cellules de defCor
    CuHostData3D<uint>      _FinalDefCor;

	///
	/// \brief _poInitCost
	/// Matrice des cellules
    sMatrixCellCost<ushort>    _poInitCost;

	///
	/// \brief optimisation Lance le calcul d'optimisation en GpGpu
	///
    void            optimisation();

private:

    void            simpleWork();

    HOST_Data2Opti  _H_data2Opt;
    DEVC_Data2Opti  _D_data2Opt;

};


#endif
