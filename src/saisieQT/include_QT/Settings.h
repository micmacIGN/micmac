#ifndef SETTINGS_H
#define SETTINGS_H

#include "Elise_QT.h"

typedef enum // Attention repercutions sur QT ... TODO à regler
{
  qNSM_GeoCube,
  qNSM_Plaquette,
  qNSM_Pts,
  qNSM_MaxLoc,
  qNSM_MinLoc,
  qNSM_NonValue
} qTypePts;

namespace Ui {
class SettingsDialog;
class HelpDialog;
}

//Min and max zoom ratio (relative)
const float GL_MAX_ZOOM = 50.f;
const float GL_MIN_ZOOM = 0.01f;

float zoomClip(float val);

//! Interface mode
enum APP_MODE { BOX2D,          /**< BOX 2D mode **/
                MASK2D,         /**< Image mask mode  **/
                MASK3D,         /**< Point cloud mask **/
                POINT2D_INIT,	/**< Points in Image (SaisieAppuisInit) - uses cAppli_SaisiePts **/
                POINT2D_PREDIC, /**< Points in Image (SaisieAppuisPredic) - uses cAppli_SaisiePts **/
                BASC            /**< 2 lines and 1 point (SaisieBasc) - uses cAppli_SaisiePts **/
};
typedef enum
{
   eEnglish = 0,
   eFrench  = 1,
   eSpanish = 2,
   /*eChinese = 3,
   eArabic  = 4,
   eRussian = 5,*/
   eNbLang
} eLANG;

typedef enum
{
   eCentroid     = 0,
   eBBoxCenter   = 1,
   eOriginCenter = 2,
   eDefault
} eSceneCenterType;

typedef enum
{
   eNavig_Ball,
   eNavig_Ball_OneTouch,
   eNavig_Orbital,
} eNavigationType;


class cParameters
{
public:
    cParameters();
    ~cParameters(){}

    //! Setters
    void setFullScreen(bool aBool)      { _fullScreen = aBool;   }
    void setPosition(QPoint pos)        { _position = pos;       }
    void setNbFen(QPoint const &aNbFen) { _nbFen = aNbFen;       }
    void setSzFen(QSize aSzFen)         { _szFen = aSzFen;       }

    void setZoomWindowValue(int aZoom)  { _zoomWindow = aZoom;   }
    void setDefPtName(QString name)     { _ptName  = name;       }
    void setPostFix(QString name)       { _postFix = name;       }

    void setLineThickness(float val)    { _lineThickness = val;  }
    void setPointDiameter(float val)    { _pointDiameter = val;  }
    void setGamma(float val)            { _gamma = val;          }
    void setForceGray(bool val)         { _forceGray = val;      }
    void setShowMasks(bool val)         { _showMasks = val;      }
    void setCenterType(int val)         { _sceneCenterType = val; }

    void setSelectionRadius(int val)    { _radius = val;         }
    void setShiftStep(float val)        {_shiftStep = val;       }

    void setPtCreationMode(qTypePts mode){ _eType = mode;        }
    void setPtCreationWindowSize(double sz){ _sz = sz;           }

    void setLanguage(int lang)          { _lang = lang;          }

    //! Getters
    bool    getFullScreen()             { return _fullScreen;    }
    QPoint  getPosition()               { return _position;      }
    QPoint  getNbFen()                  { return _nbFen;         }
    QSize   getSzFen()                  { return _szFen;         }

    float   getZoomWindowValue()        { return _zoomWindow;    }
    QString getDefPtName()              { return _ptName;        }
    QString getPostFix()                { return _postFix;       }

    float getLineThickness()            { return _lineThickness; }
    float getPointDiameter()            { return _pointDiameter; }
    float getGamma()                    { return _gamma;         }
    bool  getForceGray()                { return _forceGray;     }
    bool  getShowMasks()                { return _showMasks;     }
    int   getSceneCenterType()          { return _sceneCenterType; }

    int   getSelectionRadius()          { return _radius;        }
    float getShiftStep()                { return _shiftStep;     }

    qTypePts	   getPtCreationMode()        { return _eType;         }
    double getPtCreationWindowSize()    { return _sz;            }

    int    getLanguage()                { return _lang;          }

    //! Copy operator
    cParameters& operator =(const cParameters& params);

    //! Comparison operator
    bool operator != (cParameters &p);

    void    read();
    void    write();

    eNavigationType eNavigation() const;
    void setENavigation(const eNavigationType& eNavigation);

private:
    //! Main window parameters
    bool        _fullScreen;
    QPoint      _position;
    QPoint      _nbFen;
    QSize       _szFen;

    //! Drawing settings
    float       _lineThickness;
    float       _pointDiameter;
    float       _gamma;
    bool        _forceGray;
    bool        _showMasks;
    int         _sceneCenterType;

    //! Other parameters
    float       _zoomWindow;
    QString     _ptName;
    QString     _postFix;
    int         _radius;
    float       _shiftStep;

    //! Point creation mode
    qTypePts			_eType;
    eNavigationType _eNavigation;
    double      _sz;

    //! Language
    int         _lang;
};

//! Dialog to setup display settings
class cSettingsDlg : public QDialog
{
    Q_OBJECT

public:

    //! Default constructor
    cSettingsDlg(QWidget* parent, cParameters *params, int appMode);
    ~cSettingsDlg();

    void setParameters(cParameters &params);

    void enableMarginSpinBox(bool show = true);

signals:
    void lineThicknessChanged(float);
    void pointDiameterChanged(float);
    void gammaChanged(float);
    void forceGray(bool);
    void showMasks(bool);
    void zoomWindowChanged(float);
    void selectionRadiusChanged(int);
    void prefixTextEdit(QString);
    void shiftStepChanged(float);
    void setCenterType(int);
    void setNavigationType(int);
    void langChanged(int);

protected slots:

    void on_okButton_clicked();
    void on_cancelButton_clicked();

    //!drawing settings
    void on_LineThickness_doubleSpinBox_valueChanged(double);
    void on_PointDiameter_doubleSpinBox_valueChanged(double);
    void on_GammaDoubleSpinBox_valueChanged(double);
    void on_forceGray_checkBox_toggled(bool);
    void on_showMasks_checkBox_toggled(bool);
    void on_radioButton_centroid_toggled(bool);
    void on_radioButton_bbox_center_toggled(bool);
    void on_radioButton_origin_center_toggled(bool);
    void on_radioButtonBall_toggled(bool val);
    void on_radionButtonOrbital_toggled(bool);
    void on_radioButtonBallOneTouch_toggled(bool val);


    //!other display settings
    void on_zoomWin_spinBox_valueChanged(int);
    void on_RadiusSpinBox_valueChanged(int);
    void on_shiftStep_doubleSpinBox_valueChanged(double);
    void on_PrefixTextEdit_textChanged(QString);

    //!point creation mode
    void on_radioButtonStd_toggled(bool);
    void on_radioButtonMin_toggled(bool);
    void on_radioButtonMax_toggled(bool);
    void on_doubleSpinBoxSz_valueChanged(double);

     //!translation
    void on_comboBox_activated(int);

protected:

    //! Refreshes dialog to reflect new settings values
    void refresh();

    //! settings
    cParameters *_parameters;

    Ui::SettingsDialog* _ui;

    bool    lineItemHidden;

    int    _appMode;
};

class cHelpDlg : public QDialog
{
    Q_OBJECT

public:

    //! Default constructor
    cHelpDlg(QString title, QWidget* parent);
    ~cHelpDlg();

    void populateTableView(const QStringList &shortcuts, const QStringList &actions);

protected slots:

    void on_okButton_clicked();

protected:

    Ui::HelpDialog*         _ui;
};


#endif
