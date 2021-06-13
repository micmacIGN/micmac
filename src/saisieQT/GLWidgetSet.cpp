#include "GLWidgetSet.h"

GLWidgetSet::GLWidgetSet() :
    _widgets(0),
	_zoomWidget(NULL),
	_3DWidget(NULL)
{}

const QColor colorBG0("#323232");
const QColor colorBG1("#808080");

void GLWidgetSet::init(cParameters *params, bool modePt)
{
    int aNb = params->getNbFen().x()*params->getNbFen().y();

    if (aNb==0)
        return;

    _widgets.resize(aNb);

    _widgets[0] = new GLWidget(0, NULL);

    _pcurrentWidget = _widgets[0];

    for (int aK=1 ; aK < aNb; ++aK)
        _widgets[aK] = new GLWidget( aK, (const QGLWidget*)_widgets[0]);

    for (int aK=0 ; aK < aNb; ++aK)
    {
        _widgets[aK]->setBackgroundColors(colorBG0,colorBG1);

		if (!modePt)
			_widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );

//		qDebug() << QGuiApplication::screens()[1]->name();
//		_widgets[aK]->context()->contextHandle()->setScreen(QGuiApplication::screens()[1]);
//		qDebug() << _widgets[aK]->context()->contextHandle()->screen()->name();

		//QOpenGLContext
    }

    if (modePt)
    {
        _zoomWidget = new GLWidget(-1, (const QGLWidget*)_widgets[0]);
        _zoomWidget->setBackgroundColors(colorBG1,colorBG1);
        _zoomWidget->setContextMenuPolicy( Qt::NoContextMenu );
        _zoomWidget->setOption(cGLData::OpShow_Mess,false);
        _zoomWidget->setZoom(3.f);


        _3DWidget   = new GLWidget(10, (const QGLWidget*)_widgets[0]);
        _3DWidget->setBackgroundColors(colorBG0,colorBG1);
        _3DWidget->setContextMenuPolicy( Qt::NoContextMenu );
        _3DWidget->setOption(cGLData::OpShow_Mess,false);
    }
}

GLWidget*GLWidgetSet::currentWidget(){return _pcurrentWidget;}

int GLWidgetSet::nbWidgets() const {return _widgets.size();}

GLWidget*GLWidgetSet::zoomWidget(){return _zoomWidget;}

void GLWidgetSet::option3DPreview()
{
	threeDWidget()->setOption(cGLData::OpShow_Grid | cGLData::OpShow_Cams);
    threeDWidget()->setOption(cGLData::OpShow_Ball | cGLData::OpShow_Mess | cGLData::OpShow_BBox,false);
}

void GLWidgetSet::init3DPreview(cData* data, cParameters params)
{
	#ifdef USE_MIPMAP_HANDLER
		threeDWidget()->setGLData(new cGLData(-1, data, params));
	#else
    	threeDWidget()->setGLData(new cGLData(data,params));
	#endif
    threeDWidget()->getGLData()->setIncFirstCloud(true);
    option3DPreview();
}

void GLWidgetSet::selectCameraIn3DP(int idCam)
{
    for (int c = 0; c  < threeDWidget()->getGLData()->camerasCount(); ++c )
        threeDWidget()->getGLData()->camera(c)->setSelected(false);

    if (threeDWidget()->getGLData()->camerasCount() > idCam)
        threeDWidget()->getGLData()->camera(idCam)->setSelected(true);

    threeDWidget()->update();
}


void GLWidgetSet::widgetSetResize(int aSz)
{
    int sz = _widgets.size();

    _widgets.resize(aSz);

    for (int aK=sz ; aK < aSz; ++aK)
    {
        _widgets[aK] = new GLWidget( aK, (const QGLWidget*)_widgets[0]);

        _widgets[aK]->setBackgroundColors(colorBG0,colorBG1);
        //_widgets[aK]->setStyleSheet(style);
        //TODO: if (!modePt) _widgets[aK]->setContextMenuPolicy( Qt::NoContextMenu );
    }
}
GLWidget *GLWidgetSet::threeDWidget() const
{
    return _3DWidget;
}

GLWidgetSet::~GLWidgetSet()
{
    for (int aK=0; aK < nbWidgets();++aK) delete _widgets[aK];
    delete _zoomWidget;
}
