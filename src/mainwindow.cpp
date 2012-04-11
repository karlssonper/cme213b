#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->idCount = 0;
    this->createDatabase();
    this->connectToDataBase();

    QStringList modelsList;
    modelsList.append(QString::fromStdString("mltest_0"));
    modelsList.append(QString::fromStdString("mltest_1"));

    QStringList commonParam = this->findCommonParameters(modelsList, "sgsim");

    QStringList parameterList;
    parameterList.append("parameters$nonParamCdf$UTI_type$function");
    parameterList.append("parameters$nonParamCdf$UTI_type$special");

    parameterModel *testModel = new parameterModel(this);
    testModel->addNode(parameterList.at(0),QString("Power"),testModel->root());
    testModel->addNode(parameterList.at(1),QString("123"),testModel->root());

    //qDebug() << commonParam.at(0);


    parameterNode *test = testModel->root()->children.value("nonParamCdf");
    qDebug() << "First level name " << test->name();
    parameterNode *test1 = test->children.value("UTI_type");
    qDebug() << "Second level name " << test1->name();
    qDebug() << "Third level has children " << test1->children.count();

//    QTreeWidget *testWidget;
//    testWidget = new QTreeWidget(this);
//    testWidget->addTopLevelItem(testModel->root());

    this->ui->scrollArea->setWidget(testModel);

    //this->findCommonModels("mltest_0","sgsim",parameterList);

}

MainWindow::~MainWindow()
{

    delete ui;
}

bool MainWindow::createDatabase()
{
    this->db = QSqlDatabase::addDatabase("QSQLITE");
    db.setDatabaseName(":memory:");
    if (!db.open())
    {
        QMessageBox::critical(0, qApp->tr("Cannot open database"),
                              qApp->tr("Unable to establish a database connection.\n"
                                       "This example needs SQLite support. Please read "
                                       "the Qt SQL driver documentation for information how "
                                       "to build it.\n\n"
                                       "Click Cancel to exit."), QMessageBox::Cancel);
        return false;
    }
    else
        qDebug() << "Database Created Successfully";

    return true;
}

bool MainWindow::connectToDataBase()
{
    // Read paramters from QFile
    QDomDocument doc,doc2;
    QFile file( "/Users/lewisli/code-dev/SQLParameters/test.xml" );
    if( !file.open(QIODevice::ReadOnly ) )
        return false;
    if( !doc.setContent( &file ) )
    {
        file.close();
        return false;
    }

    this->addData(doc,"mltest_0");

    QFile file2( "/Users/lewisli/code-dev/SQLParameters/test2.xml" );
    if( !file2.open(QIODevice::ReadOnly ) )
    {
        qDebug() << "XML File not found";
        return false;
    }
    if( !doc2.setContent( &file2 ) )
    {
        file2.close();
        return false;
    }

    this->addData(doc2,"mltest_1");
    return true;

}

bool MainWindow::addData(QDomDocument &Parameters, QString propertyName)
{
    // Parse the XML Document
    QDomElement root = Parameters.documentElement();
    if( root.tagName() != "parameters" )
    {
        qDebug() << "Read improper parameter file";
        return false;
    }

    // Look for algorithm name
    QString algorithmName;

    QDomNode e = root.firstChild();

    while (!e.isNull())
    {
        QDomElement nodeElem = e.toElement();
        if (nodeElem.tagName() == QString::fromStdString("algorithm"))
        {
            algorithmName = nodeElem.attribute("name","");
            break;
        }
    }

    // Make sure we have read a real algorithm name
    bool initializeTable = false;
    if (algorithmName != "")
    {
        // Check if we already have SQL table for this algorithm
        initializeTable = !this->db.tables().contains(algorithmName);
    }

    QString *parameterStr;
    // If table doesn't exist we will need to add it
    if (initializeTable)
    {
        parameterStr =
                new QString("create table sgsim (id int primary key,");
    }

    // Add row to the SQL table
    QString *valueStr = new QString("insert into ");
    valueStr->append(algorithmName+=" ");
    valueStr->append("values(");

    // Find ID
    int currentID;
    if (!this->propertyIDHash.contains(propertyName))
    {
        // Not in hash, we need to add
        //     qDebug() << "Adding New Property To Table" << propertyName;
        this->propertyIDHash[propertyName] = this->idCount;
        currentID = this->idCount;
        this->idCount++;
    }
    else
    {
        currentID = this->propertyIDHash[propertyName];
    }

    QString idStr;
    idStr.setNum(currentID);
    idStr.append(",");
    valueStr->append(idStr);

    QStringList *parameterStrList = new QStringList();
    this->traverseNode(root.firstChild(),QString::fromStdString("parameters"),
                       valueStr,parameterStrList,initializeTable);

    // remove final comma and add a closing bracket
    valueStr->remove(valueStr->size()-1,1);
    valueStr->append(")");

    QSqlQuery query;
    // If we are creating a new table; that is this is the first time we are
    // seeing an object with its algorithm type
    if (initializeTable)
    {
        // Put name of parameters into QHash
        QString truncatedAlgoName = algorithmName.left(algorithmName.size()-1);
        this->parameterNameHashTable[truncatedAlgoName] = *parameterStrList;

        // Parse parameter names into SQL command for creating new table
        for (int i = 0; i < parameterStrList->size(); ++i)
        {
            parameterStr->append(parameterStrList->at(i));
            parameterStr->append(" varchar(255),");
        }
        parameterStr->remove(parameterStr->size()-1,1);
        parameterStr->append(")");
        query.exec(*parameterStr);
    }


    parameterStrList->size();
    this->setupModel();

    // Add row
    query.exec(*valueStr);

    // SQL doesn't allow dynamic bindings of table names; we either have to use
    // this string substituion or come up with a better database design; my
    // money is on the latter - lewisli
    QString updateStr;
    updateStr.append("UPDATE ");
    updateStr.append(algorithmName);

    if (query.prepare(updateStr.append("SET parameters$Property_Name$value = "
                                       ":propName WHERE id = :idStr")))
    {
        QString idNum;
        query.bindValue(":propName", propertyName);
        query.bindValue(":idStr", idNum.setNum(currentID));
        query.exec();
    }
    else
    {
        qDebug() << "SQL syntax error";
        return false;
    }

    // Commit changes to model
    //this->model->database().commit();
  //  this->model->submitAll();

    return true;
}

void MainWindow::setupModel()
{
//    this->model = new QSqlTableModel(this);
//    this->ui->tableView->setModel(model);
//    model->setTable(QString::fromStdString("sgsim"));
//    model->setEditStrategy(QSqlTableModel::OnManualSubmit);
//    model->select();
}

// Recursively expands the XML document
void MainWindow::traverseNode(QDomNode node, QString path, QString *values,
                              QStringList *parameterList,
                              bool initial)
{
    // Make sure node is not null
    if (node.isNull())
        return;

    QDomElement e = node.toElement();

    // Get all attributes of node
    QDomNamedNodeMap att = e.attributes();

    // Read All attirubtes
    // Note: We are using $ as seperator as _ is used as seperator in parameter
    for (int i = 0; i < att.count(); ++i)
    {
        QDomNode v = att.item(i);
        if (initial)
        {
            QString currentParameter;
            currentParameter.append(path);
            currentParameter.append("$");
            currentParameter.append(e.tagName());
            currentParameter.append("$");
            currentParameter.append(v.nodeName());
            parameterList->append(currentParameter);
        }
        values->append("'");
        values->append(v.nodeValue());
        values->append("'");
        values->append(",");
    }

    // Traverse Children
    QString childPath = path;
    childPath.append("$");
    childPath.append(e.tagName());
    this->traverseNode(node.firstChild(),childPath,values,
                       parameterList,initial);

    // Traverse Siblings
    this->traverseNode(node.nextSibling(),path,values,
                       parameterList,initial);
}

// Function to iterate through every parameter for all inputed models
// and return the common parameters
QStringList MainWindow::findCommonParameters(QStringList models,
                                             QString algoName)
{
    // It is assumed that the inputed models will be generated using
    // the same algorithm. We need a way to check this, perhaps by
    // traversing the root manager

    QStringList completeParameterList = this->parameterNameHashTable[algoName];

    // Check each parameter
    QStringList identicalParameterNames;

    for (int i = 0; i < completeParameterList.size(); ++i)
    {

        // This is a QStringList that stores the values for the current
        // parameter for each model
        QStringList parameterValue;

        for (int j = 0; j< models.size(); ++j)
        {
            QSqlQuery query;

            QString queryStr;
            queryStr.append("SELECT ");
            queryStr.append(completeParameterList.at(i));
            queryStr.append(" FROM ");
            queryStr.append(algoName);
            queryStr.append(" WHERE parameters$Property_Name$value = '");
            queryStr.append(models.at(j));
            queryStr.append("'");

            query.exec(queryStr);

            while (query.next())
            {
                parameterValue.append(query.value(0).toString());
            }
        }

        // Check if every value in parameterValue is equal
        // TODO: the following is absolutely horrendous code;
        // write this properly
        bool allIdentical = true;
        for (int a = 0; a < parameterValue.size(); ++a)
        {
            for (int b = 0; b < parameterValue.size(); ++b)
            {
                allIdentical =
                        !parameterValue.at(a).compare(parameterValue.at(b));
                if (!allIdentical)
                    break;
            }
            if (!allIdentical)
                break;
        }

        if (allIdentical)
            identicalParameterNames.append(completeParameterList.at(i));
    }

    return identicalParameterNames;
}

QStringList MainWindow::findCommonModels(QString model, QString algoName,
                                         QStringList parameterList)
{
    // First find out the values for each parameterList for inputed model
    QStringList values;
    for (int i = 0; i < parameterList.size(); ++i)
    {
        QSqlQuery query;

        QString queryStr;
        queryStr.append("SELECT ");
        queryStr.append(parameterList.at(i));
        queryStr.append(" FROM ");
        queryStr.append(algoName);
        queryStr.append(" WHERE parameters$Property_Name$value = '");
        queryStr.append(model);
        queryStr.append("'");

        query.exec(queryStr);
        //qDebug() << queryStr;

        while (query.next())
            values.append(query.value(0).toString());
    }

    // Now we search our table for models that match ALL given parameters
    QStringList commonModels;
    QSqlQuery query;

    QString queryStr;
    queryStr.append("SELECT parameters$Property_Name$value FROM ");
    queryStr.append(algoName);
    queryStr.append( " WHERE ");

    for (int i = 0; i < parameterList.size(); ++i)
    {
        queryStr.append(parameterList.at(i));
        queryStr.append(" = '");
        queryStr.append(values.at(i));
        queryStr.append("'");

        if (i+1<parameterList.size())
        {
            queryStr.append(" AND ");
        }

    }

    query.exec(queryStr);

    while (query.next())
        commonModels.append(query.value(0).toString());

    return commonModels;
}


parameterModel::parameterModel(QWidget *parent) : QTreeWidget(parent)
{
    this->root_ = new parameterNode("parameters","");
    this->root_->parent = 0;
    this->addTopLevelItem(this->root());
    this->setColumnCount(2);
}

void parameterModel::addNode(QString pathName, QString value, parameterNode *parent)
{
    // Current parent
    parameterNode *currentParent;

    // Parse the path name
    int tokenIndex = pathName.indexOf("$");

    // Check if current path level is equal to parent
    QString parentPathLevel = pathName.left(tokenIndex);
    if(parent->name() != parentPathLevel)
    {
        qDebug() << "parent mismatch";
        return;
    }

    // Truncate path
    QString truncatedPath = pathName.right(pathName.size()-tokenIndex-1);

    // Get current level
    tokenIndex = truncatedPath.indexOf("$");

    QString currentPathLevel = truncatedPath.left(tokenIndex);

    // We are only writing node to leaf, and empty elsewhere
    QString valueStr;

    if (tokenIndex == -1)
        valueStr = value;

    // Check to see if is a child exists
    if (parent->children.contains(currentPathLevel))
    {
        // If the level already exists we simply pass down that branch
        currentParent = parent->children.value(currentPathLevel);
    }
    else
    {
        // If level doesn't exist, we need to create a new node here
        parameterNode *newNode = new parameterNode(currentPathLevel,valueStr,parent);
        newNode->parent = parent;
        parent->children.insert(currentPathLevel,newNode);
        parent->addChild(newNode);
        currentParent = newNode;
    }

    // Stop recursion if we have reached last token
    if (tokenIndex == -1)
        return;

    // Recurse
    this->addNode(truncatedPath,value,currentParent);
}
