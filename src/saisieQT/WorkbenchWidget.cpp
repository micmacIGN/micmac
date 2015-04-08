#include "WorkbenchWidget.h"
#include "ui_WorkbenchWidget.h"

cWorkBenchWidget::cWorkBenchWidget(QWidget *parent) :
	QWidget(parent),
	ui(new Ui::cWorkBenchWidget)
{
	ui->setupUi(this);
}

cWorkBenchWidget::~cWorkBenchWidget()
{
	delete ui;
}
