#include "GpGpu/GpGpu_InterOptimisation.h"

#include "GpGpu/SData2Optimize.h"

extern "C" void TestOptimisationOneDirectionZ(Data2Optimiz<CuDeviceData3D> &d2O);

int main()
{
    cout << "Launch Data optimisation GpGpu ***" << endl;

    // Declaration des variables du cote du HOST
    HOST_Data2Opti h2O;
    // Declaration des variables du cote du DEVICE
    DEVC_Data2Opti d2O;

    uint nbLines    = 1;
    uint lLines     = 2;
    uint depth      = NAPPEMAX;

    short2 dZ = make_short2(-depth/2,depth/2);

    uint sizeMaxLine = (uint)(1.5f*sqrt((float)lLines * lLines + nbLines * nbLines));

    h2O.ReallocParam(sizeMaxLine);
    d2O.ReallocParam(sizeMaxLine);

    uint pitIdStream = WARPSIZE;
    uint pitStream   = NAPPEMAX;

    // param

    for (uint p= 0 ; p < nbLines; p++)
    {
        uint lenghtLine = lLines;

        h2O.SetParamLine(p,pitStream,pitIdStream,lenghtLine);

        uint sizeStreamLine = lLines * depth;

        pitIdStream += iDivUp(lenghtLine,       WARPSIZE) * WARPSIZE;
        pitStream   += iDivUp(sizeStreamLine,   WARPSIZE) * WARPSIZE;
    }

    h2O.ReallocInputIf(pitStream + NAPPEMAX,pitIdStream + WARPSIZE);
    h2O._s_InitCostVol.Fill(0);

    // index
    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lLines; aK++)
        {
            h2O._s_Index[h2O._param[0][idLine].y + aK ]= dZ;

            uint idStrm = h2O._param[0][idLine].x + pitStrm - dZ.x;

            for ( int aPx = dZ.x ; aPx < dZ.y; aPx++)
                h2O._s_InitCostVol[idStrm + aPx]  = 10000 * (idLine + 1) + (aK+1) * 1000 + aPx - dZ.x + 1;

            pitStrm += depth;
        }
    }

    //h2O._s_InitCostVol.OutputValues();

    h2O.SetNbLine(nbLines);    
    d2O.SetNbLine(h2O._nbLines);

    h2O.ReallocOutputIf(h2O._s_InitCostVol.GetSize());

    h2O._s_ForceCostVol[0].Fill(0);

    d2O.ReallocIf(h2O);


    //      Transfert des données vers le device                            ---------------		-
    d2O.CopyHostToDevice(h2O);
    d2O._s_ForceCostVol[0].CopyHostToDevice(h2O._s_ForceCostVol[0].pData());

    TestOptimisationOneDirectionZ(d2O);

    d2O.CopyDevicetoHost(h2O);

    h2O._s_InitCostVol.OutputValues();

    //h2O._s_Index.OutputValues(0,XY,NEGARECT,3,make_short2(0,0));

    h2O._s_ForceCostVol[0].OutputValues();
    //

    return 0;
}

