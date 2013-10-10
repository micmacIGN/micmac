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

    uint nbLines    = 16;
    uint lLines     = 312;
    uint depth      = NAPPEMAX;

    short2 dZ   = make_short2(-depth/2,depth/2);
    depth       = count(dZ);

    uint sizeMaxLine = (uint)(1.5f*sqrt((float)lLines * lLines + nbLines * nbLines));

    h2O.ReallocParam(sizeMaxLine);
    d2O.ReallocParam(sizeMaxLine);

    uint pit_Strm_DZ    = WARPSIZE;
    uint pit_Strm_ICost = NAPPEMAX;

    // param

    for (uint p= 0 ; p < nbLines; p++)
    {
        uint lenghtLine = lLines;

        h2O.SetParamLine(p,pit_Strm_ICost,pit_Strm_DZ,lenghtLine);

        uint sizeStreamLine = lLines * depth;

        pit_Strm_DZ     += iDivUp(lenghtLine,       WARPSIZE) * WARPSIZE;
        pit_Strm_ICost  += iDivUp(sizeStreamLine,   WARPSIZE) * WARPSIZE;
    }

    h2O.ReallocInputIf(pit_Strm_ICost + NAPPEMAX,pit_Strm_DZ + WARPSIZE);
    h2O._s_InitCostVol.Fill(0);

    // index
    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lLines; aK++)
        {
            h2O._s_Index[h2O._param[0][idLine].y + aK ] = dZ;

            uint idStrm = h2O._param[0][idLine].x + pitStrm - dZ.x;

            for ( int aPx = dZ.x ; aPx < dZ.y; aPx++)
                h2O._s_InitCostVol[idStrm + aPx]  = 10000 * (idLine + 1) + (aK+1) * 1000 + aPx - dZ.x + 1;

            pitStrm += depth;
        }
    }

    h2O.SetNbLine(nbLines);    
    d2O.SetNbLine(h2O._nbLines);

    h2O.ReallocOutputIf(h2O._s_InitCostVol.GetSize());

    h2O._s_ForceCostVol[0].Fill(0);

    d2O.ReallocIf(h2O);

    //      Transfert des données vers le device                            ---------------		-
    d2O.CopyHostToDevice(h2O);
    d2O._s_ForceCostVol[0].CopyHostToDevice(h2O._s_ForceCostVol[0].pData());

    //h2O._s_InitCostVol.OutputValues();

    TestOptimisationOneDirectionZ(d2O);

    d2O.CopyDevicetoHost(h2O);

    //h2O._s_ForceCostVol[0].OutputValues();

    //
    uint errorCount = 0;

    for (uint idLine= 0 ; idLine < nbLines; idLine++)
    {
        uint    pitStrm = 0;

        for (uint aK= 0 ; aK < lLines; aK++)
        {
            short2 dZ = h2O._s_Index[h2O._param[0][idLine].y + aK ];

            uint idStrm = h2O._param[0][idLine].x + pitStrm - dZ.x;

            for ( int aPx = dZ.x ; aPx < dZ.y; aPx++)
                if( h2O._s_InitCostVol[idStrm + aPx]  != h2O._s_ForceCostVol[0][idStrm + aPx])
                    errorCount++;

            pitStrm += depth;
        }
    }

    printf("\nError Count   = %d/%d\n",errorCount,h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX);
    printf("Error percent = %f\n",(((float)errorCount*100)/(h2O._s_InitCostVol.GetSize()- 2*NAPPEMAX)));

    return 0;
}

