#!/usr/bin/env wolframscript
(* Run the TypeI C++ solver for Konishi at g=1/10, export converged solution as JSON *)

SetOptions[$Output, FormatType->OutputForm];
ToString1 = ToString[#, PageWidth->Infinity]&;

(* Paths *)
execPath = "/Users/dz1614/Documents/Working/QSC_Numerics/reference/qsc/local operators N4 SYM/core/";
pertFile = "/Users/dz1614/Documents/Working/QSC_Numerics/reference/qsc/local operators N4 SYM/data/perturbative/perturbative_data_Delta02_b10_b20_f11_f21_f31_f41_a10_a20_sol1.mx";
outputFile = "/Users/dz1614/Documents/Working/QSC_Numerics/tests/fixtures/konishi_converged_g01.json";

(* Load perturbative data *)
Get[pertFile];

(* Konishi quantum numbers *)
nb[1]=0; nb[2]=0; nf[1]=1; nf[2]=1; nf[3]=1; nf[4]=1; na[1]=0; na[2]=0;
sol = 1;

(* Derived quantities *)
L = (nf[1]+nf[2]+nf[3]+nf[4]+na[1]+na[2]-nb[1]-nb[2])/2;
\[CapitalDelta]0 = nf[1]/2+nf[2]/2+nf[3]/2+nf[4]/2+na[1]+na[2];
\[Lambda]0[a_] := nf[a]+{2,1,0,-1}[[a]];
\[CapitalLambda] = (1-\[Lambda]0[1]-\[Lambda]0[4])/2;
\[Lambda][a_] := \[Lambda]0[a]+\[CapitalLambda];

(* Hyperparameters *)
cutP = 16;
nPoints = cutP + 2;
cutQai = 24;
QaiShift = 60;
\[Epsilon] = 10^(-30);
WP = 186;
errorTol = 24;
precssf = 2*(errorTol-4);
maxiter = 5;

gloop = 1/10;

(* Build perturbative initial guess *)
DeltaNum = \[CapitalDelta]0 + Sum[\[CapitalDelta]g[k]*gloop^k, {k,2,12,2}] /. sbWeak;
cCoeff[a_,n_] := Sum[cg[a,n,k]*gloop^k, {k,-4,12}] /. sbWeak /. cg[__]->0;

paramVec = Join[
  {N[gloop,WP], N[DeltaNum,WP]},
  Table[N[cCoeff[1,n],WP], {n,2,cutP,2}],
  Table[N[cCoeff[2,n],WP], {n,2,cutP,2}],
  Table[N[cCoeff[3,n],WP], {n,2,cutP,2}],
  Table[N[cCoeff[4,n],WP], {n,2,cutP,2}]
];

init11 = SetPrecision[paramVec, WP];
g0 = init11[[1]];
\[Gamma]0 = init11[[2]] - \[CapitalDelta]0;
Mt = Table[\[Lambda][a], {a,4}];

(* Denormalize c-vectors for C++ *)
c1 = 1/g0^Mt[[1]]*Table[init11[[2+j]], {j,1,cutP/2}];
c2 = 1/g0^Mt[[2]]*Table[init11[[2+cutP/2+j]], {j,1,cutP/2}];
c3 = 1/g0^Mt[[3]]*Table[init11[[2+2*cutP/2+j]], {j,1,cutP/2}];
c4 = 1/g0^Mt[[4]]*Table[init11[[2+3*cutP/2+j]], {j,1,cutP/2}];
c = {Insert[c1,1,1], Insert[c2,1,1], Insert[c3,1,1], Insert[c4,1,1]};
convertionFactor = Table[1/g0^Mt[[a]], {a,4}, cutP/2] // Flatten;

(* Build C++ command-line arguments *)
ingen = Table["", {k,1,24+4*cutP/2}];
For[k=0, k<=cutP/2, k++,
  ingen[[21+k]] = ToString1[CForm[Im[c[[1,k+1]]]]];
  ingen[[22+cutP/2+k]] = ToString1[CForm[Re[c[[2,k+1]]]]];
  ingen[[23+2*cutP/2+k]] = ToString1[CForm[Im[c[[3,k+1]]]]];
  ingen[[24+3*cutP/2+k]] = ToString1[CForm[Re[c[[4,k+1]]]]];
];
ingen[[1]] = ToString1[WP];
ingen[[2]] = ToString1[cutP/2];
ingen[[3]] = ToString1[cutQai/2];
ingen[[4]] = ToString1[QaiShift];
ingen[[5]] = ToString1[nPoints];
ingen[[6]] = ToString1[-Round[Log10[\[Epsilon]]]];
ingen[[7]] = ToString1[precssf];
ingen[[8]] = ToString1[errorTol];
ingen[[9]] = ToString1[maxiter];
ingen[[10]] = ToString1[nb[1]];
ingen[[11]] = ToString1[nb[2]];
ingen[[12]] = ToString1[nf[1]];
ingen[[13]] = ToString1[nf[2]];
ingen[[14]] = ToString1[nf[3]];
ingen[[15]] = ToString1[nf[4]];
ingen[[16]] = ToString1[na[1]];
ingen[[17]] = ToString1[na[2]];
ingen[[18]] = ToString1[sol];
ingen[[19]] = ToString1[NumberForm[Re[N[g0,{WP,WP}]],{WP,WP}]];
ingen[[20]] = ToString1[CForm[Re[N[\[Gamma]0,{WP,WP}]]]];

(* Run C++ *)
cmd = "\"" <> execPath <> "TypeI_exec.out\" " <> StringRiffle[ingen, " "];
Print["Running C++ solver..."];
Print["Command length: ", StringLength[cmd], " chars"];

output = Import["!" <> cmd, "String"];

(* Parse output *)
Toexpr = ToExpression[output];

If[Toexpr[[1]] === 0,
  Print["SUCCESS - converged"];,
  Print["WARNING - did not fully converge (code ", Toexpr[[1]], ")"];
];

(* Extract converged solution *)
(* Toexpr[[4]] = final Gamma (anomalous dimension) *)
(* Toexpr[[5-8]] = c-vectors (denormalized) *)
(* Toexpr[[9]] = {iterations, dDelta, ...} *)

GammaFinal = Toexpr[[4]];
DeltaFinal = \[CapitalDelta]0 + GammaFinal;
c1out = Drop[Toexpr[[5]], 1] / convertionFactor[[1;;cutP/2]];
c2out = Drop[Toexpr[[6]], 1] / convertionFactor[[cutP/2+1;;cutP]];
c3out = Drop[Toexpr[[7]], 1] / convertionFactor[[cutP+1;;3*cutP/2]];
c4out = Drop[Toexpr[[8]], 1] / convertionFactor[[3*cutP/2+1;;2*cutP]];

Print["Delta_final = ", N[DeltaFinal, 20]];
Print["Iterations: ", Toexpr[[9,1]]];

(* Build converged parameter vector in same format as input — take Re to drop numerical noise *)
convergedVec = Re @ Join[
  {N[gloop, 30], N[DeltaFinal, 30]},
  N[c1out, 30],
  N[c2out, 30],
  N[c3out, 30],
  N[c4out, 30]
];

(* Also export initial guess for comparison *)
result = <|
  "g" -> N[gloop, 20],
  "Delta0" -> \[CapitalDelta]0,
  "quantum_numbers" -> {nb[1],nb[2],nf[1],nf[2],nf[3],nf[4],na[1],na[2],sol},
  "cutP" -> cutP,
  "nPoints" -> nPoints,
  "cutQai" -> cutQai,
  "QaiShift" -> QaiShift,
  "Mt" -> Re[N[Mt, 20]],
  "converged_params" -> convergedVec,
  "initial_params" -> Re[N[paramVec, 30]],
  "Delta_converged" -> Re[N[DeltaFinal, 20]],
  "Delta_initial" -> Re[N[DeltaNum, 20]],
  "iterations" -> Toexpr[[9,1]],
  "error_code" -> Toexpr[[1]]
|>;

Export[outputFile, result, "JSON"];
Print["Exported to: ", outputFile];
