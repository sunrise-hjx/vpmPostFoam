dictionary vpmPostDict;
const word dictName("vpmPostDict");

IOobject dictIO
(
    dictName, // name of the file
    mesh.time().system(), // path to where the file is
    mesh, // reference to the mesh needed by the constructor
    IOobject::READ_IF_PRESENT // indicate that reading this dictionary is compulsory
);

// Initialise the dictionary object
vpmPostDict = IOdictionary(dictIO);

// This template method needs to know the type of the variable and can provide
// a default value if the entry is not found in the dictionary
bool Q_Output( vpmPostDict.lookupOrDefault<bool>("Q", false) );
bool Omega_Output( vpmPostDict.lookupOrDefault<bool>("Omega", false) );
bool U_Output( vpmPostDict.lookupOrDefault<bool>("U", false) );
label Dimension( vpmPostDict.lookupOrDefault<label>("dimension", 3) );

if (!Q_Output && !Omega_Output && !U_Output)
{
    Info << "No fields." << nl << nl << "end" << nl;
    exit(1);
}