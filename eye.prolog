% --------------------------------------------------
% Euler Yet another proof Engine - EYE -- Jos De Roo
% --------------------------------------------------

% See https://github.com/josd/eye


:- if(current_prolog_flag(dialect, swi)).
:- if(current_prolog_flag(version_data, swi(6, _, _, _))).
:- style_check(-atom).
:- endif.
:- initialization(catch(set_prolog_stack(local, limit(2^33)), _, true)).
:- initialization(catch(set_prolog_stack(global, limit(2^35)), _, true)).
:- initialization(set_prolog_flag(agc_margin, 10000000)).
:- endif.


:- use_module(library(lists)).
:- use_module(library(gensym)).
:- use_module(library(system)).
:- use_module(library(terms)).
:- use_module(library('url.pl')).
:- use_module(library(charsio)).
:- if(current_prolog_flag(dialect, swi)).
:- use_module(library(when), [when/2]).
:- use_module(library(qsave)).
:- catch(use_module(library(base64)), _, true).
:- catch(use_module(library(process)), _, true).
:- catch(use_module(library(sha)), _, true).
:- catch(use_module(library(uri)), _, true).
:- endif.
:- if(\+current_predicate(date_time_stamp/2)).
:- load_foreign_files(['pl-tai'], [], install).
:- endif.


:- if(current_predicate(set_stream/2)).
:- initialization(catch(set_stream(user_output, encoding(utf8)), _, true)).
:- else.
:- set_prolog_flag(encoding, utf8).
:- endif.


version_info('EYE rel. v17.0217.1257 josd').


license_info('MIT License

Copyright (c) 2009 Jos De Roo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.').


help_info('Usage: eye <options>* <data>* <query>*
eye
	swipl -x eye.pvm --
<options>
	--curl-http-header <field>	to pass HTTP header <field> to curl
	--debug				output debug info on stderr
	--debug-cnt			output debug info about counters on stderr
	--debug-djiti			output debug info about DJITI on stderr
	--debug-pvm			output debug info about PVM code on stderr
	--help				show help info
	--hmac-key <key>		HMAC key
	--ignore-inference-fuse		do not halt in case of inference fuse
	--ignore-syntax-error		do not halt in case of syntax error
	--image <pvm-file>		output all <data> and all code to <pvm-file>
	--license			show license info
	--multi-query			query answer loop
	--n3p				output all <data> as N3 P-code on stdout
	--no-distinct-input		no distinct triples in the input
	--no-distinct-output		no distinct answers in the output
	--no-genid			no generated id in Skolem IRI
	--no-numerals			no numerals in the output
	--no-qnames			no qnames in the output
	--no-qvars			no qvars in the output
	--no-skolem <prefix>		no uris with <prefix> in the output
	--nope				no proof explanation
	--pass-all-ground		ground the rules and run --pass-all
	--pass-only-new			output only new derived triples
	--pass-turtle			output the --turtle data
	--probe				output speedtest info on stderr
	--profile			output profile info on stderr
	--rule-histogram		output rule histogram info on stderr
	--statistics			output statistics info on stderr
	--streaming-reasoning		streaming reasoning on --turtle data
	--strict			strict mode
	--strings			output log:outputString objects on stdout
	--tactic existing-path		Euler path using homomorphism
	--tactic limited-answer <count>	give only a limited number of answers
	--tactic limited-brake <count>	take only a limited number of brakes
	--tactic limited-step <count>	take only a limited number of steps
	--tactic linear-select		select each rule only once
	--traditional			traditional mode
	--version			show version info
	--warn				output warning info on stderr
	--wcache <uri> <file>		to tell that <uri> is cached as <file>
<data>
	<n3-data>			N3 triples and rules
	--plugin <n3p-data>		N3 P-code
	--proof <n3-proof>		N3 proof
	--turtle <ttl-data>		Turtle data
<query>
	--pass				output deductive closure
	--pass-all			output deductive closure plus rules
	--query <n3-query>		output filtered with filter rules').


:- dynamic(answer/7).		% answer(Predicate, Subject, Object, Subject_index, Object_index, Subject_arg_1, Object_arg_1)
:- dynamic(argi/1).
:- dynamic(base_uri/1).
:- dynamic(bcnd/2).
:- dynamic(bgot/3).
:- dynamic(brake/0).
:- dynamic(bref/2).
:- dynamic(bvar/1).
:- dynamic(cpred/1).
:- dynamic(evar/3).
:- dynamic(exopred/3).		% exopred(Predicate, Subject, Object)
:- dynamic(fact/1).
:- dynamic(flag/1).
:- dynamic(flag/2).
:- dynamic(got_dq/0).
:- dynamic(got_head/0).
:- dynamic(got_labelvars/2).
:- dynamic(got_pi/0).
:- dynamic(got_random/3).
:- dynamic(got_sq/0).
:- dynamic(got_unique/2).
:- dynamic(got_wi/5).
:- dynamic(graph/2).
:- dynamic(hash_value/2).
:- dynamic(implies/3).		% implies(Premise, Conclusion, Source)
:- dynamic(input_statements/1).
:- dynamic(intern/1).
:- dynamic(keep_skolem/1).
:- dynamic(keywords/1).
:- dynamic(lemma/6).		% lemma(Count, Source, Premise, Conclusion, Premise-Conclusion_index, Rule)
:- dynamic(mtime/2).
:- dynamic(ncllit/0).
:- dynamic(ns/2).
:- dynamic(pfx/2).
:- dynamic(pred/1).
:- dynamic(preda/1).
:- dynamic(prfstep/8).		% prfstep(Conclusion_triple, Conclusion_triple_index, Premise, Premise_index, Conclusion, Rule, Chaining, Source)
:- dynamic(qevar/3).
:- dynamic(query/2).
:- dynamic(quvar/3).
:- dynamic(rule_uvar/1).
:- dynamic(scope/1).
:- dynamic(scount/1).
:- dynamic(semantics/2).
:- dynamic(span/1).
:- dynamic(table/3).
:- dynamic(tmpfile/1).
:- dynamic(tuple/2).
:- dynamic(tuple/3).
:- dynamic(tuple/4).
:- dynamic(tuple/5).
:- dynamic(tuple/6).
:- dynamic(tuple/7).
:- dynamic(tuple/8).
:- dynamic(wcache/2).
:- dynamic(wpfx/1).
:- dynamic(wtcache/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#mu>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#pi>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#sigma>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#reflexive>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tactic>'/2).
:- dynamic('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'/2).
:- dynamic('<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>'/2).
:- dynamic('<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>'/2).
:- dynamic('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'/2).
:- dynamic('<http://www.w3.org/2000/10/swap/list#in>'/2).
:- dynamic('<http://www.w3.org/2000/10/swap/list#member>'/2).
:- dynamic('<http://www.w3.org/2000/10/swap/log#implies>'/2).
:- dynamic('<http://www.w3.org/2000/10/swap/log#outputString>'/2).
:- dynamic('<http://www.w3.org/2002/07/owl#sameAs>'/2).


% Main goal

main :-
	current_prolog_flag(argv, Argv),
	(	append(_, ['--'|Argvp], Argv)
	->	true
	;	Argvp = Argv
	),
	(	Argvp = ['-']
	->	catch((read_line_to_codes(user_input, Ac), atom_codes(As, Ac)), _, As = ''),
		(	As = ''
		->	Argvs = []
		;	atomic_list_concat(Argvs, ' ', As)
		)
	;	Argvs = Argvp
	),
	argv(Argvs, Argus),
	findall(Argij,
		(	argi(Argij)
		),
		Argil
	),
	append(Argil, Argi),
	format(user_error, 'eye~@~@~n', [w0(Argi), w1(Argus)]),
	flush_output(user_error),
	(	memberchk('--no-genid', Argus)
	->	Vns = 'http://eulersharp.sourceforge.net/.well-known/genid/#'
	;	Run1 is random(2^30)*random(2^30)*random(2^30)*random(2^30),
		atom_number(Run2, Run1),
		catch(sha_hash(Run2, Run3, [algorithm(sha1)]), _,
			(	format(user_error, '** ERROR ** EYE requires swipl package clib which can be installed from http://www.swi-prolog.org/Download.html~n', []),
				flush_output(user_error),
				halt(1)
			)
		),
		atom_codes(Run4, Run3),
		base64xml(Run4, Run5),
		atomic_list_concat(['http://eulersharp.sourceforge.net/.well-known/genid/', Run5, '#'], Vns)
	),
	nb_setval(var_ns, Vns),
	version_info(Version),
	format(user_error, '~w~n', [Version]),
	flush_output(user_error),
	prolog_flag(version, PVersion),
	format(user_error, '~w~n', [PVersion]),
	flush_output(user_error),
	catch(process_create(path(curl), ['--version'], [stdin(null), stdout(null), stderr(null)]), _,
		(	format(user_error, '** WARNING ** EYE depends on curl which can be installed from https://curl.haxx.se/download.html **~n', []),
			flush_output(user_error)
		)
	),
	catch(process_create(path(cturtle), [], [stdin(null), stdout(null), stderr(null)]), _,
		(	format(user_error, '** WARNING ** EYE depends on cturtle which can be installed from https://github.com/melgi/cturtle/releases/ **~n', []),
			flush_output(user_error)
		)
	),
	(	retract(prolog_file_type(qlf, qlf))
	->	assertz(prolog_file_type(pvm, qlf))
	;	true
	),
	(	memberchk('--profile', Argus)
	->	(	current_predicate(profon/0)
		->	yap_flag(profiling, on),
			profinit,
			profon
		;	profiler(_, cputime)
		)
	;	true
	),
	catch(gre(Argus), Exc,
		(	Exc = halt
		->	true
		;	format(user_error, '** ERROR ** gre ** ~w~n', [Exc]),
			flush_output(user_error),
			nb_setval(exit_code, 1)
		)
	),
	(	memberchk('--profile', Argus)
	->	(	current_predicate(profon/0)
		->	profoff,
			showprofres,
			format(user_error, '~n', []),
			flush_output(user_error)
		;	profiler(_, false),
			tell(user_error),
			(	current_predicate(show_profile/2)
			->	show_profile(plain, 25)
			;	show_profile([top(-1)])
			),
			told
		)
	;	true
	),
	(	flag(statistics)
	->	statistics
	;	true
	),
	(	flag('debug-pvm')
	->	tell(user_error),
		ignore(vm_list(_))
	;	true
	),
	(	flag('debug-djiti')
	->	forall(
			(	pred(Pred)
			),
			(	(	P =.. [Pred, _, _],
					predicate_property(P, indexed(Ind2))
				->	format(user_error, 'DJITI ~w/2 indexed ~w~n', [Pred, Ind2])
				;	true
				),
				(	P =.. [Pred, _, _, _, _, _, _],
					predicate_property(P, indexed(Ind6))
				->	format(user_error, 'DJITI ~w/6 indexed ~w~n', [Pred, Ind6])
				;	true
				),
				(	P =.. [Pred, _, _, _, _, _, _, _],
					predicate_property(P, indexed(Ind7))
				->	format(user_error, 'DJITI ~w/7 indexed ~w~n', [Pred, Ind7])
				;	true
				)
			)
		),
		(	predicate_property(type_index(_, _, _), indexed(Indt3))
		->	format(user_error, 'DJITI type_index/3 indexed ~w~n', [Indt3])
		;	true
		),
		(	predicate_property(implies(_, _, _), indexed(Indi3))
		->	format(user_error, 'DJITI implies/3 indexed ~w~n', [Indi3])
		;	true
		),
		(	predicate_property(lemma(_, _, _, _, _, _), indexed(Indl6))
		->	format(user_error, 'DJITI lemma/6 indexed ~w~n', [Indl6])
		;	true
		),
		(	predicate_property(prfstep(_, _, _, _, _, _, _, _), indexed(Indp8))
		->	format(user_error, 'DJITI prfstep/8 indexed ~w~n', [Indp8])
		;	true
		),
		format(user_error, '~n', []),
		flush_output(user_error)
	;	true
	),
	nb_getval(exit_code, EC),
	flush_output,
	halt(EC).


argv([], []) :-
	!.
argv([Arg|Argvs], [U, V|Argus]) :-
	sub_atom(Arg, B, 1, E, '='),
	sub_atom(Arg, 0, B, _, U),
	memberchk(U, ['--curl-http-header', '--hmac-key', '--image', '--no-skolem', '--plugin', '--proof', '--query', '--tactic', '--turtle',
		      '--brake', '--step', '--tmp-file', '--tquery', '--trules', '--wget-path', '--yabc']),	% DEPRECATED
	!,
	sub_atom(Arg, _, E, 0, V),
	argv(Argvs, Argus).
argv([Arg|Argvs], [Arg|Argus]) :-
	argv(Argvs, Argus).



% ---------------------------------------------------------------
% GRE (Generic Reasoning Engine) supporting Explainable Reasoning
% ---------------------------------------------------------------

gre(Argus) :-
	statistics(runtime, [T0, _]),
	statistics(walltime, [T1, _]),
	format(user_error, 'starting ~w [msec cputime] ~w [msec walltime]~n', [T0, T1]),
	flush_output(user_error),
	nb_getval(var_ns, Vns),
	nb_setval(exit_code, 0),
	nb_setval(indentation, 0),
	nb_setval(limit, -1),
	nb_setval(bnet, not_done),
	nb_setval(fnet, not_done),
	nb_setval(table, -1),
	nb_setval(tuple, -1),
	nb_setval(fdepth, 0),
	nb_setval(pdepth, 0),
	nb_setval(cdepth, 0),
	(	input_statements(Ist)
	->	nb_setval(input_statements, Ist)
	;	nb_setval(input_statements, 0)
	),
	opts(Argus, Args),
	(	\+flag('multi-query'),
		Args = []
	->	opts(['--help'], _)
	;	true
	),
	(	(	flag('no-qvars')
		;	flag('pass-all-ground')
		)
	->	atomic_list_concat(['<', Vns, '>'], Vpfx),
		retractall(pfx('var:', _)),
		assertz(pfx('var:', Vpfx))
	;	true
	),
	(	flag(n3p)
	->	format(':- style_check(-discontiguous).~n', []),
		format(':- style_check(-singleton).~n', []),
		format(':- multifile(exopred/3).~n', []),
		format(':- multifile(implies/3).~n', []),
		format(':- multifile(pfx/2).~n', []),
		format(':- multifile(pred/1).~n', []),
		format(':- multifile(prfstep/8).~n', []),
		format(':- multifile(scope/1).~n', []),
		format(':- multifile(scount/1).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#mu>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#pi>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/fl-rules#sigma>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#reflexive>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tactic>\'/2).~n', []),
		format(':- multifile(\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/2000/10/swap/log#implies>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/2000/10/swap/log#outputString>\'/2).~n', []),
		format(':- multifile(\'<http://www.w3.org/2002/07/owl#sameAs>\'/2).~n', []),
		format('flag(\'no-skolem\', \'~w\').~n', [Vns])
	;	true
	),
	args(Args),
	(	implies(_, Conc, _),
		(	var(Conc)
		;	Conc \= answer(_, _, _, _, _, _, _),
			Conc \= cn([answer(_, _, _, _, _, _, _)|_])
		)
	->	true
	;	(	\+flag(image, _),
			\+flag(tactic, 'linear-select')
		->	assertz(flag(tactic, 'linear-select'))
		;	true
		)
	),
	findall(Sc,
		(	scope(Sc)
		),
		Scope
	),
	nb_setval(scope, Scope),
	statistics(runtime, [_, T2]),
	statistics(walltime, [_, T3]),
	(	flag('streaming-reasoning')
	->	true
	;	format(user_error, 'networking ~w [msec cputime] ~w [msec walltime]~n', [T2, T3]),
		flush_output(user_error)
	),
	nb_getval(input_statements, SC),
	(	flag(image, File)
	->	assertz(argi(Argus)),
		retractall(flag(image, _)),
		assertz(flag('no-skolem', Vns)),
		retractall(input_statements(_)),
		assertz(input_statements(SC)),
		reset_gensym,
		(	current_predicate(qsave:qsave_program/1)
		->	qsave_program(File)
		;	save_program(File)
		),
		throw(halt)
	;	true
	),
	(	flag(n3p)
	->	(	SC =\= 0
		->	write(scount(SC)),
			writeln('.')
		;	true
		),
		writeln('end_of_file.'),
		throw(halt)
	;	true
	),
	(	\+implies(_, answer(_, _, _, _, _, _, _), _),
		\+implies(_, cn([answer(_, _, _, _, _, _, _)|_]), _),
		\+query(_, _),
		\+flag('pass-only-new'),
		\+flag('multi-query'),
		\+flag(strings)
	->	throw(halt)
	;	true
	),
	(	flag(strings)
	->	true
	;	version_info(Version),
		format('#Processed by ~w~n', [Version]),
		findall(Argij,
			(	argi(Argij)
			),
			Argil
		),
		append(Argil, Argi),
		format('#eye~@~@~n~n', [w0(Argi), w1(Argus)]),
		flush_output
	),
	(	flag(nope)
	->	true
	;	(	pfx('r:', _)
		->	true
		;	assertz(pfx('r:', '<http://www.w3.org/2000/10/swap/reason#>'))
		),
		(	\+flag(traditional)
		->	true
		;	(	pfx('var:', _)
			->	true
			;	atomic_list_concat(['<', Vns, '>'], Vpfx),
				assertz(pfx('var:', Vpfx))
			),
			(	pfx('n3:', _)
			->	true
			;	assertz(pfx('n3:', '<http://www.w3.org/2004/06/rei#>'))
			)
		)
	),
	(	flag('pass-only-new')
	->	wh
	;	true
	),
	nb_setval(tr, 0),
	nb_setval(tc, 0),
	nb_setval(tp, 0),
	nb_setval(wn, 0),
	nb_setval(rn, 0),
	nb_setval(lemma_count, 0),
	nb_setval(lemma_cursor, 0),
	nb_setval(output_statements, 0),
	nb_setval(answer_count, 0),
	(	flag('multi-query')
	->	nb_setval(mq, 0),
		tmp_file(Tmp),
		assertz(flag('tmp-file', Tmp)),	% DEPRECATED
		repeat,
		catch((read_line_to_codes(user_input, Fc), atom_codes(Fa, Fc)), _, Fa = end_of_file),
		(	atomic_list_concat([Fi, Fo], ',', Fa)
		->	open(Fo, write, Fos, [encoding(utf8)])
		;	Fi = Fa,
			Fos = user_output
		),
		(	Fi = end_of_file
		->	true
		;	statistics(walltime, [_, _]),
			nb_getval(output_statements, Outb),
			statistics(inferences, Infb),
			catch(args(['--query', Fi]), Exc1,
				(	format(user_error, '** ERROR ** args ** ~w~n', [Exc1]),
					flush_output(user_error),
					nb_setval(exit_code, 1)
				)
			),
			tell(Fos),
			catch(eam(0), Exc2,
				(	(	Exc2 = halt
					->	true
					;	format(user_error, '** ERROR ** eam ** ~w~n', [Exc2]),
						flush_output(user_error),
						nb_setval(exit_code, 1)
					)
				)
			),
			(	flag(strings)
			->	wst
			;	true
			),
			forall(
				(	retract(preda(Pa))
				),
				(	Ans =.. [Pa, _, _, _, _, _, _, answer],
					retractall(Ans)
				)
			),
			forall(
				(	answer(A1, A2, A3, A4, A5, A6, A7),
					nonvar(A1)
				),
				(	retract(answer(A1, A2, A3, A4, A5, A6, A7))
				)
			),
			retractall(implies(_, answer(_, _, _, _, _, _, _), _)),
			retractall(implies(_, cn([answer(_, _, _, _, _, _, _)|_]), _)),
			retractall(query(_, _)),
			retractall(prfstep(answer(_, _, _, _, _, _, _), _, _, _, _, _, _, _)),
			retractall(lemma(_, _, _, _, _, _)),
			retractall(got_wi(_, _, _, _, _)),
			retractall(wpfx(_)),
			retractall('<http://www.w3.org/2000/10/swap/log#outputString>'(_, _)),
			retractall('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>'(_, _)),
			nb_setval(csv_header, []),
			cnt(mq),
			nb_getval(mq, Cnt),
			(	Cnt mod 10000 =:= 0
			->	garbage_collect_atoms
			;	true
			),
			statistics(runtime, [_, Ti4]),
			statistics(walltime, [_, Ti5]),
			format(user_error, 'reasoning ~w [msec cputime] ~w [msec walltime]~n', [Ti4, Ti5]),
			flush_output(user_error),
			nb_getval(output_statements, Oute),
			Outd is Oute-Outb,
			catch(Outs is round(Outd/Ti5*1000), _, Outs = ''),
			(	flag(strings)
			->	nl
			;	format('#DONE ~3d [sec] mq=~w out=~d out/sec=~w~n~n', [Ti5, Cnt, Outd, Outs])
			),
			timestamp(Stmp),
			statistics(inferences, Infe),
			Infd is Infe-Infb,
			catch(Infs is round(Infd/Ti5*1000), _, Infs = ''),
			format(user_error, '~w mq=~w out=~d inf=~w sec=~3d out/sec=~w inf/sec=~w~n~n', [Stmp, Cnt, Outd, Infd, Ti5, Outs, Infs]),
			flush_output(user_error),
			told,
			fail
		)
	;	catch(eam(0), Exc3,
			(	(	Exc3 = halt
				->	true
				;	format(user_error, '** ERROR ** eam ** ~w~n', [Exc3]),
					flush_output(user_error),
					nb_setval(exit_code, 1)
				)
			)
		)
	),
	(	flag(strings)
	->	wst
	;	true
	),
	nb_getval(tc, TC),
	nb_getval(tp, TP),
	flush_output(user_error),
	statistics(runtime, [Cpu, T4]),
	statistics(walltime, [_, T5]),
	(	\+flag('multi-query')
	->	format(user_error, 'reasoning ~w [msec cputime] ~w [msec walltime]~n', [T4, T5]),
		flush_output(user_error)
	;	true
	),
	nb_getval(input_statements, Inp),
	nb_getval(output_statements, Outp),
	timestamp(Stamp),
	Ent is TC,
	Step is TP,
	nb_getval(tr, TR),
	Brake is TR,
	(	statistics(inferences, Inf)
	->	true
	;	Inf = ''
	),
	catch(Speed is round(Inf/Cpu*1000), _, Speed = ''),
	(	flag(strings)
	->	true
	;	format('#~w in=~d out=~d ent=~d step=~w brake=~w inf=~w sec=~3d inf/sec=~w~n#ENDS~n~n', [Stamp, Inp, Outp, Ent, Step, Brake, Inf, Cpu, Speed])
	),
	format(user_error, '~w in=~d out=~d ent=~d step=~w brake=~w inf=~w sec=~3d inf/sec=~w~n~n', [Stamp, Inp, Outp, Ent, Step, Brake, Inf, Cpu, Speed]),
	flush_output(user_error),
	(	flag('rule-histogram')
	->	findall([RTC, RTP, R],
			(	table(ETP, tp, Rule),
				nb_getval(ETP, RTP),
				(	table(ETC, tc, Rule)
				->	nb_getval(ETC, RTC)
				;	RTC = 0
				),
				with_output_to(atom(R), wt(Rule))
			),
			CntRl
		),
		sort(CntRl, CntRs),
		reverse(CntRs, CntRr),
		format(user_error, '>>> rule histogram TR=~w <<<~n', [TR]),
		forall(
			(	member(RCnt, CntRr)
			),
			(	(	last(RCnt, '<http://www.w3.org/2000/10/swap/log#implies>'(X, Y)),
					cn_conj(X, XC),
					c_append(XC, pstep(_), Z),
					catch(clause(Y, Z), _, fail)
				->	format(user_error, 'TC=~w TP=~w for component ~w~n', RCnt)
				;	format(user_error, 'TC=~w TP=~w for rule ~w~n', RCnt)
				)
			)
		),
		format(user_error, '~n', []),
		flush_output(user_error)
	;	true
	).


% command line options

opts([], []) :-
	!.
% DEPRECATED
opts(['--ances'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--ances']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--brake', Lim|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--brake', '--tactic limited-brake']),
	flush_output(user_error),
	(	number(Lim)
	->	Limit = Lim
	;	catch(atom_number(Lim, Limit), Exc,
			(	format(user_error, '** ERROR ** brake ** ~w~n', [Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	),
	retractall(flag(brake, _)),
	assertz(flag(brake, Limit)),
	opts(Argus, Args).
opts(['--curl-http-header', Field|Argus], Args) :-
	!,
	assertz(flag('curl-http-header', Field)),
	opts(Argus, Args).
opts(['--debug'|Argus], Args) :-
	!,
	retractall(flag(debug)),
	assertz(flag(debug)),
	opts(Argus, Args).
opts(['--debug-cnt'|Argus], Args) :-
	!,
	retractall(flag('debug-cnt')),
	assertz(flag('debug-cnt')),
	opts(Argus, Args).
opts(['--debug-djiti'|Argus], Args) :-
	!,
	retractall(flag('debug-djiti')),
	assertz(flag('debug-djiti')),
	opts(Argus, Args).
opts(['--debug-pvm'|Argus], Args) :-
	!,
	retractall(flag('debug-pvm')),
	assertz(flag('debug-pvm')),
	opts(Argus, Args).
opts(['--help'|_], _) :-
	\+flag(image, _),
	\+flag('debug-pvm'),
	!,
	help_info(Help),
	format(user_error, '~w~n', [Help]),
	flush_output(user_error),
	throw(halt).
opts(['--hmac-key', Key|Argus], Args) :-
	!,
	retractall(flag('hmac-key', _)),
	assertz(flag('hmac-key', Key)),
	opts(Argus, Args).
opts(['--ignore-inference-fuse'|Argus], Args) :-
	!,
	retractall(flag('ignore-inference-fuse')),
	assertz(flag('ignore-inference-fuse')),
	opts(Argus, Args).
opts(['--ignore-syntax-error'|Argus], Args) :-
	!,
	retractall(flag('ignore-syntax-error')),
	assertz(flag('ignore-syntax-error')),
	opts(Argus, Args).
opts(['--image', File|Argus], Args) :-
	!,
	retractall(flag(image, _)),
	assertz(flag(image, File)),
	opts(Argus, Args).
% DEPRECATED
opts(['--kgb'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--kgb']),
	flush_output(user_error),
	opts(Argus, Args).
opts(['--license'|_], _) :-
	!,
	license_info(License),
	format(user_error, '~w~n', [License]),
	flush_output(user_error),
	throw(halt).
opts(['--multi-query'|Argus], Args) :-
	!,
	retractall(flag('multi-query')),
	assertz(flag('multi-query')),
	opts(Argus, Args).
opts(['--n3p'|Argus], Args) :-
	!,
	retractall(flag(n3p)),
	assertz(flag(n3p)),
	opts(Argus, Args).
% DEPRECATED
opts(['--no-blank'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--no-blank']),
	flush_output(user_error),
	retractall(flag('no-blank')),
	assertz(flag('no-blank')),
	opts(Argus, Args).
% DEPRECATED
opts(['--no-branch'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--no-branch']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--no-distinct'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--no-distinct', 'no-distinct-output']),
	flush_output(user_error),
	retractall(flag('no-distinct-output')),
	assertz(flag('no-distinct-output')),
	opts(Argus, Args).
opts(['--no-distinct-input'|Argus], Args) :-
	!,
	retractall(flag('no-distinct-input')),
	assertz(flag('no-distinct-input')),
	opts(Argus, Args).
opts(['--no-distinct-output'|Argus], Args) :-
	!,
	retractall(flag('no-distinct-output')),
	assertz(flag('no-distinct-output')),
	opts(Argus, Args).
opts(['--no-genid'|Argus], Args) :-
	!,
	retractall(flag('no-genid')),
	assertz(flag('no-genid')),
	opts(Argus, Args).
opts(['--no-numerals'|Argus], Args) :-
	!,
	retractall(flag('no-numerals')),
	assertz(flag('no-numerals')),
	opts(Argus, Args).
opts(['--no-qnames'|Argus], Args) :-
	!,
	retractall(flag('no-qnames')),
	assertz(flag('no-qnames')),
	opts(Argus, Args).
opts(['--no-qvars'|Argus], Args) :-
	!,
	retractall(flag('no-qvars')),
	assertz(flag('no-qvars')),
	opts(Argus, Args).
opts(['--no-skolem', Prefix|Argus], Args) :-
	!,
	assertz(flag('no-skolem', Prefix)),
	opts(Argus, Args).
% DEPRECATED
opts(['--no-span'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--no-span']),
	flush_output(user_error),
	retractall(flag('no-span')),
	assertz(flag('no-span')),
	opts(Argus, Args).
opts(['--nope'|Argus], Args) :-
	!,
	retractall(flag(nope)),
	assertz(flag(nope)),
	opts(Argus, Args).
opts(['--pass-all-ground'|Argus], Args) :-
	!,
	retractall(flag('pass-all-ground')),
	assertz(flag('pass-all-ground')),
	opts(['--pass-all'|Argus], Args).
opts(['--pass-only-new'|Argus], Args) :-
	!,
	retractall(flag('pass-only-new')),
	assertz(flag('pass-only-new')),
	opts(Argus, Args).
opts(['--pass-turtle'|Argus], Args) :-
	!,
	retractall(flag('pass-turtle')),
	assertz(flag('pass-turtle')),
	opts(Argus, Args).
% DEPRECATED
opts(['--pcl'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--pcl', '--n3p']),
	flush_output(user_error),
	retractall(flag(n3p)),
	assertz(flag(n3p)),
	opts(Argus, Args).
opts(['--probe'|_], _) :-
	!,
	probe,
	throw(halt).
opts(['--profile'|Argus], Args) :-
	!,
	retractall(flag(profile)),
	assertz(flag(profile)),
	opts(Argus, Args).
% DEPRECATED
opts(['--quiet'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--quiet']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--quick-false'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--quick-false']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--quick-possible'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--quick-possible']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--quick-answer'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--quick-answer', '--tactic limited-answer 1']),
	retractall(flag('limited-answer', _)),
	assertz(flag('limited-answer', 1)),
	opts(Argus, Args).
opts(['--rule-histogram'|Argus], Args) :-
	!,
	retractall(flag('rule-histogram')),
	assertz(flag('rule-histogram')),
	opts(Argus, Args).
% DEPRECATED
opts(['--single-answer'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--single-answer', '--tactic limited-answer 1']),
	flush_output(user_error),
	retractall(flag('limited-answer', _)),
	assertz(flag('limited-answer', 1)),
	opts(Argus, Args).
opts(['--statistics'|Argus], Args) :-
	!,
	retractall(flag(statistics)),
	assertz(flag(statistics)),
	opts(Argus, Args).
% DEPRECATED
opts(['--step', Lim|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--step', '--tactic limited-step']),
	flush_output(user_error),
	(	number(Lim)
	->	Limit = Lim
	;	catch(atom_number(Lim, Limit), Exc,
			(	format(user_error, '** ERROR ** step ** ~w~n', [Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	),
	retractall(flag(step, _)),
	assertz(flag(step, Limit)),
	opts(Argus, Args).
opts(['--streaming-reasoning'|Argus], Args) :-
	!,
	retractall(flag('streaming-reasoning')),
	assertz(flag('streaming-reasoning')),
	opts(Argus, Args).
opts(['--strict'|Argus], Args) :-
	!,
	retractall(flag(strict)),
	assertz(flag(strict)),
	opts(Argus, Args).
opts(['--strings'|Argus], Args) :-
	!,
	retractall(flag(strings)),
	assertz(flag(strings)),
	opts(Argus, Args).
opts(['--tactic', 'existing-path'|Argus], Args) :-
	!,
	retractall(flag(tactic, 'existing-path')),
	assertz(flag(tactic, 'existing-path')),
	opts(Argus, Args).
opts(['--tactic', 'limited-answer', Lim|Argus], Args) :-
	!,
	(	number(Lim)
	->	Limit = Lim
	;	catch(atom_number(Lim, Limit), Exc,
			(	format(user_error, '** ERROR ** limited-answer ** ~w~n', [Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	),
	retractall(flag('limited-answer', _)),
	assertz(flag('limited-answer', Limit)),
	opts(Argus, Args).
opts(['--tactic', 'limited-brake', Lim|Argus], Args) :-
	!,
	(	number(Lim)
	->	Limit = Lim
	;	catch(atom_number(Lim, Limit), Exc,
			(	format(user_error, '** ERROR ** limited-brake ** ~w~n', [Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	),
	retractall(flag('limited-brake', _)),
	assertz(flag('limited-brake', Limit)),
	opts(Argus, Args).
opts(['--tactic', 'limited-step', Lim|Argus], Args) :-
	!,
	(	number(Lim)
	->	Limit = Lim
	;	catch(atom_number(Lim, Limit), Exc,
			(	format(user_error, '** ERROR ** limited-step ** ~w~n', [Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	),
	retractall(flag('limited-step', _)),
	assertz(flag('limited-step', Limit)),
	opts(Argus, Args).
opts(['--tactic', 'linear-select'|Argus], Args) :-
	!,
	retractall(flag(tactic, 'linear-select')),
	assertz(flag(tactic, 'linear-select')),
	opts(Argus, Args).
% DEPRECATED
opts(['--tactic', 'single-answer'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED and is now ~w~n', ['--tactic single-answer', '--tactic limited-answer 1']),
	flush_output(user_error),
	retractall(flag('limited-answer', _)),
	assertz(flag('limited-answer', 1)),
	opts(Argus, Args).
opts(['--tactic', Tactic|_], _) :-
	!,
	throw(not_supported_tactic(Tactic)).
% DEPRECATED
opts(['--think'|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--think']),
	retractall(flag(think)),
	assertz(flag(think)),
	opts(Argus, Args).
% DEPRECATED
opts(['--tmp-file', File|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--tmp-file']),
	flush_output(user_error),
	retractall(flag('tmp-file', _)),
	assertz(flag('tmp-file', File)),
	opts(Argus, Args).
opts(['--traditional'|Argus], Args) :-
	!,
	retractall(flag(traditional)),
	assertz(flag(traditional)),
	opts(Argus, Args).
opts(['--version'|_], _) :-
	!,
	throw(halt).
opts(['--warn'|Argus], Args) :-
	!,
	retractall(flag(warn)),
	assertz(flag(warn)),
	opts(Argus, Args).
opts(['--wcache', Argument, File|Argus], Args) :-
	!,
	absolute_uri(Argument, Arg),
	retractall(wcache(Arg, _)),
	assertz(wcache(Arg, File)),
	opts(Argus, Args).
% DEPRECATED
opts(['--wget-path', _|Argus], Args) :-
	!,
	format(user_error, '** WARNING ** option ~w is DEPRECATED~n', ['--wget-path']),
	flush_output(user_error),
	opts(Argus, Args).
% DEPRECATED
opts(['--yabc', File|Argus], Args) :-
	!,
	retractall(flag(image, _)),
	assertz(flag(image, File)),
	opts(Argus, Args).
opts([Arg|_], _) :-
	\+memberchk(Arg, ['--help', '--pass', '--pass-all', '--plugin', '--proof', '--query', '--turtle']),
	\+memberchk(Arg, ['--tquery', '--trules']),	% DEPRECATED
	sub_atom(Arg, 0, 2, _, '--'),
	!,
	throw(not_supported_option(Arg)).
opts([Arg|Argus], [Arg|Args]) :-
	opts(Argus, Args).


probe :-
	tmp_file(File),
	(	curl_http_headers(Headers),
		atomic_list_concat(['curl -s -L -H "Accept: text/plain" ', Headers, 'http://www.agfa.com/w3c/temp/graph-100000.n3p -o ', File], Cmd),
		catch(exec(Cmd, _), _, fail)
	->	statistics(walltime, [_, T1]),
		S1 is 100000000/T1
	;	open(File, write, Out, [encoding(utf8)]),
		tell(Out),
		format(':- style_check(-discontiguous).~n', []),
		format(':- style_check(-singleton).~n', []),
		(	between(0, 99, I),
			format(':- dynamic(\'<http://eulersharp.sourceforge.net/2007/07test/graph#i~d>\'/2).~n', [I]),
			format('pred(\'<http://eulersharp.sourceforge.net/2007/07test/graph#i~d>\').~n', [I]),
			fail
		;	true
		),
		(	between(1, 100000, _),
			S is random(10000),
			P is random(100),
			O is random(10000),
			format('\'<http://eulersharp.sourceforge.net/2007/07test/graph#i~d>\'(\'<http://eulersharp.sourceforge.net/2007/07test/graph#i~d>\',
				\'<http://eulersharp.sourceforge.net/2007/07test/graph#i~d>\').~n', [P, S, O]),
			fail
		;	true
		),
		told,
		statistics(walltime, [_, _]),
		S1 is 0
	),
	open(File, read, In, [encoding(utf8)]),
	repeat,
	read_term(In, Rt, []),
	(	Rt = end_of_file
	->	true
	;	(	Rt = ':-'(Rg)
		->	call(Rg)
		;	(	call(Rt)
			->	true
			;	djitis(Rt)
			)
		),
		fail
	),
	statistics(walltime, [_, T2]),
	S2 is 100000000/T2,
	statistics(runtime, [_, _]),
	(	between(1, 100, _),
		forall(
			(	pred(P)
			),
			(	forall(
					(	call(P, _, _)
					),
					(	true
					)
				)
			)
		),
		fail
	;	true
	),
	statistics(runtime, [_, T3]),
	S3 is 10000000000/T3,
	timestamp(Stamp),
	format(user_error, '~w web-triples/sec=~0f file-triples/sec=~0f memory-triples/sec=~0f~n~n', [Stamp, S1, S2, S3]),
	flush_output(user_error),
	close(In),
	delete_file(File).


curl_http_headers(Headers) :-
	findall(Header,
		(	flag('curl-http-header', Field),
			atomic_list_concat(['-H "', Field, '" '], Header)
		),
		List
	),
	atomic_list_concat(List, Headers).


args([]) :-
	!.
args(['--pass'|Args]) :-
	!,
	(	flag(nope),
		\+flag('limited-answer', _),
		(	flag('no-distinct-input')
		->	flag('no-distinct-output')
		;	true
		),
		\+implies(_, answer(_, _, _, _, _, _, _), _),
		\+implies(_, cn([answer(_, _, _, _, _, _, _)|_]), _)
	->	assertz(query(exopred(P, S, O), exopred(P, S, O)))
	;	assertz(implies(exopred(P, S, O), answer(P, S, O, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass>'))
	),
	(	flag(n3p)
	->	portray_clause(implies(exopred(P, S, O), answer(P, S, O, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass>'))
	;	true
	),
	args(Args).
args(['--pass-all'|Args]) :-
	!,
	assertz(implies(cn([exopred(P, S, O), '<http://www.w3.org/2000/10/swap/log#notEqualTo>'(P, '<http://www.w3.org/2000/10/swap/log#implies>')]),
			answer(P, S, O, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>')),
	assertz(implies(cn(['<http://www.w3.org/2000/10/swap/log#implies>'(A, C), '<http://www.w3.org/2000/10/swap/log#notEqualTo>'(A, true)]),
			answer('<http://www.w3.org/2000/10/swap/log#implies>', A, C, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>')),
	assertz(implies(':-'(C, A),
			answer(':-', C, A, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>')),
	(	flag(n3p)
	->	portray_clause(implies(cn([exopred(P, S, O), '<http://www.w3.org/2000/10/swap/log#notEqualTo>'(P, '<http://www.w3.org/2000/10/swap/log#implies>')]),
			answer(P, S, O, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>')),
		portray_clause(implies(cn(['<http://www.w3.org/2000/10/swap/log#implies>'(A, C), '<http://www.w3.org/2000/10/swap/log#notEqualTo>'(A, true)]),
			answer('<http://www.w3.org/2000/10/swap/log#implies>', A, C, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>')),
		portray_clause(implies(cn([':-'(C, A), '<http://www.w3.org/2000/10/swap/log#notEqualTo>'(A, true)]),
			answer(':-', C, A, _, _, _, _), '<http://eulersharp.sourceforge.net/2003/03swap/pass-all>'))
	;	true
	),
	args(Args).
args(['--plugin', Argument|Args]) :-
	!,
	absolute_uri(Argument, Arg),
	(	wcacher(Arg, File)
	->	format(user_error, 'GET ~w FROM ~w ', [Arg, File]),
		flush_output(user_error)
	;	format(user_error, 'GET ~w ', [Arg]),
		flush_output(user_error),
		(	(	sub_atom(Arg, 0, 5, _, 'http:')
			->	true
			;	sub_atom(Arg, 0, 6, _, 'https:')
			)
		->	(	flag('tmp-file', File)	% DEPRECATED
			->	true
			;	tmp_file(File),
				assertz(tmpfile(File))
			),
			curl_http_headers(Headers),
			atomic_list_concat(['curl -s -L -H "Accept: text/plain" ', Headers, '"', Arg, '" -o ', File], Cmd),
			catch(exec(Cmd, _), Exc,
				(	format(user_error, '** ERROR ** ~w ** ~w~n', [Arg, Exc]),
					flush_output(user_error),
					(	retract(tmpfile(File))
					->	delete_file(File)
					;	true
					),
					flush_output,
					halt(1)
				)
			)
		;	(	sub_atom(Arg, 0, 5, _, 'file:')
			->	parse_url(Arg, Parts),
				memberchk(path(File), Parts)
			;	File = Arg
			)
		)
	),
	(	File = '-'
	->	In = user_input
	;	open(File, read, In, [encoding(utf8)])
	),
	repeat,
	read_term(In, Rt, []),
	(	Rt = end_of_file
	->	catch(read_line_to_codes(In, _), _, true)
	;	n3pin(Rt, In, File),
		fail
	),
	!,
	(	File = '-'
	->	true
	;	close(In)
	),
	(	retract(tmpfile(File))
	->	delete_file(File)
	;	true
	),
	findall(SCnt,
		(	retract(scount(SCnt))
		),
		SCnts
	),
	sum(SCnts, SC),
	nb_getval(input_statements, IN),
	Inp is SC+IN,
	nb_setval(input_statements, Inp),
	format(user_error, 'SC=~w~n', [SC]),
	flush_output(user_error),
	args(Args).
args(['--proof', Arg|Args]) :-
	!,
	absolute_uri(Arg, A),
	atomic_list_concat(['<', A, '>'], R),
	assertz(scope(R)),
	(	flag(n3p)
	->	portray_clause(scope(R))
	;	true
	),
	n3_n3p(Arg, data),
	(	got_pi
	->	true
	;	assertz(implies(cn(['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'(S, '<http://www.w3.org/2000/10/swap/reason#Inference>'),
				'<http://www.w3.org/2000/10/swap/reason#gives>'(S, G)]),
				G, '<http://eulersharp.sourceforge.net/2003/03swap/proof-lemma>')),
		assertz(implies(cn(['<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'(S, '<http://www.w3.org/2000/10/swap/reason#Extraction>'),
				'<http://www.w3.org/2000/10/swap/reason#gives>'(S, G)]),
				G, '<http://eulersharp.sourceforge.net/2003/03swap/proof-lemma>')),
		assertz(got_pi)
	),
	args(Args).
args(['--query', Arg|Args]) :-
	!,
	n3_n3p(Arg, query),
	args(Args).
% DEPRECATED
args(['--tquery', Arg|Args]) :-
	!,
	assertz(flag(tquery)),
	n3_n3p(Arg, tquery),
	args(Args).
% DEPRECATED
args(['--trules', Arg|Args]) :-
	!,
	absolute_uri(Arg, A),
	atomic_list_concat(['<', A, '>'], R),
	assertz(scope(R)),
	(	flag(n3p)
	->	portray_clause(scope(R))
	;	true
	),
	n3_n3p(Arg, trules),
	args(Args).
args(['--turtle', Argument|Args]) :-
	!,
	absolute_uri(Argument, Arg),
	(	wcacher(Arg, File)
	->	format(user_error, 'GET ~w FROM ~w ', [Arg, File]),
		flush_output(user_error)
	;	format(user_error, 'GET ~w ', [Arg]),
		flush_output(user_error),
		(	(	sub_atom(Arg, 0, 5, _, 'http:')
			->	true
			;	sub_atom(Arg, 0, 6, _, 'https:')
			)
		->	(	flag('tmp-file', File)	% DEPRECATED
			->	true
			;	tmp_file(File),
				assertz(tmpfile(File))
			),
			curl_http_headers(Headers),
			atomic_list_concat(['curl -s -L -H "Accept: text/turtle" ', Headers, '"', Arg, '" -o ', File], Cmd),
			catch(exec(Cmd, _), Exc,
				(	format(user_error, '** ERROR ** ~w ** ~w~n', [Arg, Exc]),
					flush_output(user_error),
					(	retract(tmpfile(File))
					->	delete_file(File)
					;	true
					),
					flush_output,
					halt(1)
				)
			)
		;	(	sub_atom(Arg, 0, 5, _, 'file:')
			->	parse_url(Arg, Parts),
				memberchk(path(File), Parts)
			;	File = Arg
			)
		)
	),
	atomic_list_concat(['-b=', Arg], Base),
	(	flag('pass-turtle')
	->	catch(process_create(path(cturtle), ['-f=nt', Base, file(File)], [stdout(std), stderr(std)]), Exc,
			(	format(user_error, '** ERROR ** ~w ** ~w~n', [Arg, Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		)
	;	(	flag(strict)
		->	Format = '-f=n3p-rdiv'
		;	Format = '-f=n3p'
		),
		catch(process_create(path(cturtle), [Format, Base, file(File)], [stdout(pipe(In)), stderr(std)]), Exc,
			(	format(user_error, '** ERROR ** ~w ** ~w~n', [Arg, Exc]),
				flush_output(user_error),
				flush_output,
				halt(1)
			)
		),
		nb_setval(wn, 0),
		nb_setval(rn, 0),
		nb_setval(sc, 0),
		nb_setval(tc, 0),
		nb_setval(tp, 0),
		nb_setval(tr, 0),
		set_stream(In, encoding(utf8)),
		repeat,
		read_term(In, Rt, []),
		(	Rt = end_of_file
		->	catch(read_line_to_codes(In, _), _, true)
		;	(	flag('streaming-reasoning')
			->	(	Rt \= ':-'(_),
					Rt \= flag(_, _),
					Rt \= scope(_),
					Rt \= pfx(_, _),
					Rt \= pred(_),
					Rt \= cpred(_),
					Rt \= scount(_)
				->	Rt =.. [P, S, O],
					implies(Prem, Conc, _),
					(	(	Prem = exopred(P, S, O)
						;	Prem = Rt
						)
					->	true
					;	(	Prem = cn([exopred(P, S, O)|U])
						;	Prem = cn([Rt|U])
						),
						clist(U, V),
						call(V)
					),
					(	ground(Conc)
					->	true
					;	nb_getval(wn, W),
						labelvars(Conc, W, N, skolem),
						nb_setval(wn, N)
					),
					(	Conc = cn(C)
					->	forall(
							(	member(Q, C)
							),
							(	(	Q = exopred(X, Y, Z)
								->	Qt =.. [X, Y, Z]
								;	Qt = Q
								),
								functor(Qt, F, _),
								(	pred(F)
								->	true
								;	assertz(pred(F)),
									format(':- dynamic(~q).~n', [F/2]),
									format(':- multifile(~q).~n', [F/2]),
									format('pred(~q).~n', [F])
								),
								format('~q.~n', [Qt])
							)
						),
						nb_getval(sc, I),
						length(C, J),
						K is I+J,
						nb_setval(sc, K)
					;	(	Conc = exopred(X, Y, Z)
						->	Qt =.. [X, Y, Z]
						;	Qt = Conc
						),
						functor(Qt, F, _),
						(	pred(F)
						->	true
						;	assertz(pred(F)),
							format(':- dynamic(~q).~n', [F/2]),
							format(':- multifile(~q).~n', [F/2]),
							format('pred(~q).~n', [F])
						),
						format('~q.~n', [Qt]),
						cnt(sc)
					)
				;	(	Rt = pred(F)
					->	(	pred(F)
						->	true
						;	assertz(pred(F))
						)
					;	true
					),
					(	Rt = scount(SCount)
					->	assertz(scount(SCount))
					;	true
					),
					format('~q.~n', [Rt])
				)
			;	n3pin(Rt, In, File)
			),
			fail
		),
		!,
		(	File = '-'
		->	true
		;	close(In)
		),
		(	retract(tmpfile(File))
		->	delete_file(File)
		;	true
		),
		findall(SCnt,
			(	retract(scount(SCnt))
			),
			SCnts
		),
		sum(SCnts, SC),
		nb_getval(input_statements, IN),
		Inp is SC+IN,
		nb_setval(input_statements, Inp),
		format(user_error, 'SC=~w~n', [SC]),
		flush_output(user_error),
		(	flag('streaming-reasoning')
		->	timestamp(Stamp),
			statistics(runtime, [Cpu, Wall]),
			nb_getval(sc, Outp),
			nb_getval(tc, TC),
			Ent is TC,
			nb_getval(tp, TP),
			Step is TP,
			nb_getval(tr, TR),
			Brake is TR,
			statistics(inferences, Inf),
			catch(Rate is round(Outp/Wall*1000), _, Rate = ''),
			catch(Speed is round(Inf/Cpu*1000), _, Speed = ''),
			format('~q.~n', [scount(Outp)]),
			format(user_error, 'streaming-reasoning ~w [msec cputime] ~w [msec walltime] (~w triples/s)~n', [Cpu, Wall, Rate]),
			format(user_error, '~w in=~d out=~d ent=~d step=~w brake=~w inf=~w sec=~3d inf/sec=~w~n~n', [Stamp, Inp, Outp, Ent, Step, Brake, Inf, Cpu, Speed]),
			flush_output(user_error)
		;	true
		)
	),
	args(Args).
args([Arg|Args]) :-
	absolute_uri(Arg, A),
	atomic_list_concat(['<', A, '>'], R),
	assertz(scope(R)),
	(	flag(n3p)
	->	portray_clause(scope(R))
	;	true
	),
	n3_n3p(Arg, data),
	args(Args).


n3pin(Rt, In, File) :-
	(	Rt = ':-'(Rg)
	->	call(Rg),
		(	flag(n3p)
		->	format('~q.~n', [Rt])
		;	true
		)
	;	(	(	Rt = ':-'(Rh, _)
			->	predicate_property(Rh, dynamic)
			;	predicate_property(Rt, dynamic)
			)
		->	true
		;	(	File = '-'
			->	true
			;	close(In)
			),
			(	retract(tmpfile(File))
			->	delete_file(File)
			;	true
			),
			throw(builtin_redefinition(Rt))
		),
		(	Rt = pfx(Pfx, _)
		->	retractall(pfx(Pfx, _))
		;	true
		),
		(	Rt = scope(Scope)
		->	nb_setval(current_scope, Scope)
		;	true
		),
		(	Rt = ':-'(Ci, Pi)
		->	(	Ci = true
			->	call(Pi)
			;	nb_getval(current_scope, Si),
				copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(Pi, Ci), Ri),
				cn_conj(Pi, Pn),
				(	flag(nope)
				->	Ph = Pn
				;	(	Pi = when(Ai, Bi)
					->	c_append(Bi, istep(Si, Pi, Ci, Ri), Bh),
						Ph = when(Ai, Bh)
					;	c_append(Pn, istep(Si, Pi, Ci, Ri), Ph)
					)
				),
				(	flag('rule-histogram')
				->	(	Ph = when(Ak, Bk)
					->	c_append(Bk, pstep(Ri), Bj),
						Pj = when(Ak, Bj)
					;	c_append(Ph, pstep(Ri), Pj)
					)
				;	Pj = Ph
				),
				functor(Ci, CPi, _),
				(	flag(n3p)
				->	portray_clause(cpred(CPi)),
					portray_clause(':-'(Ci, Pn))
				;	(	\+cpred(CPi)
					->	assertz(cpred(CPi))
					;	true
					),
					assertz(':-'(Ci, Pj))
				)
			)
		;	(	Rt \= implies(_, _, _),
				Rt \= scount(_),
				\+flag('no-distinct-input'),
				call(Rt)
			->	true
			;	(	Rt \= pred('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>')
				->	djitis(Rt),
					(	flag(n3p),
						Rt \= scount(_)
					->	format('~q.~n', [Rt])
					;	true
					)
				;	true
				),
				(	Rt \= flag(_, _),
					Rt \= scope(_),
					Rt \= pfx(_, _),
					Rt \= pred(_),
					Rt \= cpred(_),
					Rt \= scount(_)
				->	(	flag(nope)
					->	true
					;	term_index(Rt, Rnd),
						nb_getval(current_scope, Src),
						(	\+prfstep(Rt, Rnd, true, _, Rt, _, forward, Src)
						->	assertz(prfstep(Rt, Rnd, true, _, Rt, _, forward, Src))
						;	true
						)
					)
				;	true
				)
			)
		)
	).


% N3 to N3P compiler

n3_n3p(Argument, Mode) :-
	absolute_uri(Argument, Arg),
	(	flag('tmp-file', Tmp)	% DEPRECATED
	->	true
	;	tmp_file(Tmp)
	),
	(	flag('ignore-syntax-error')
	->	Ise = 'IGNORED'
	;	Ise = 'ERROR'
	),
	(	wcacher(Arg, File)
	->	format(user_error, 'GET ~w FROM ~w ', [Arg, File]),
		flush_output(user_error)
	;	format(user_error, 'GET ~w ', [Arg]),
		flush_output(user_error),
		(	(	sub_atom(Arg, 0, 5, _, 'http:')
			->	true
			;	sub_atom(Arg, 0, 6, _, 'https:')
			)
		->	File = Tmp,
			(	flag('tmp-file', _)	% DEPRECATED
			->	true
			;	assertz(tmpfile(File))
			),
			curl_http_headers(Headers),
			atomic_list_concat(['curl -s -L -H "Accept: text/n3, text/turtle" ', Headers, '"', Arg, '" -o ', File], Cmd),
			catch(exec(Cmd, _), Exc1,
				(	format(user_error, '** ERROR ** ~w ** ~w~n', [Arg, Exc1]),
					flush_output(user_error),
					(	retract(tmpfile(File))
					->	delete_file(File)
					;	true
					),
					flush_output,
					halt(1)
				)
			)
		;	(	sub_atom(Arg, 0, 5, _, 'file:')
			->	(	parse_url(Arg, Parts)
				->	memberchk(path(File), Parts)
				;	sub_atom(Arg, 7, _, 0, File)
				)
			;	File = Arg
			)
		)
	),
	(	File = '-'
	->	In = user_input
	;	open(File, read, In, [encoding(utf8)])
	),
	retractall(base_uri(_)),
	(	Arg = '-'
	->	absolute_uri('', Abu),
		assertz(base_uri(Abu))
	;	assertz(base_uri(Arg))
	),
	retractall(ns(_, _)),
	(	Arg = '-'
	->	D = '#'
	;	atomic_list_concat([Arg, '#'], D)
	),
	assertz(ns('', D)),
	retractall(keywords(_)),
	retractall(quvar(_, _, _)),
	retractall(qevar(_, _, _)),
	retractall(evar(_, _, _)),
	nb_setval(line_number, 1),
	nb_setval(sc, 0),
	nb_setval(semantics, []),
	nb_setval(smod, true),
	atomic_list_concat(['\'<', Arg, '>\''], Src),
	atomic_list_concat([Tmp, '_p'], Tmp_p),
	assertz(tmpfile(Tmp_p)),
	open(Tmp_p, write, Ws, [encoding(utf8)]),
	tell(Ws),
	catch(
		(	repeat,
			tokens(In, Tokens),
			document(Triples, Tokens, Rest),
			(	Rest = []
			->	true
			;	nb_getval(line_number, Ln),
				throw(invalid_document(after_line(Ln)))
			),
			(	Mode = semantics
			->	nb_getval(semantics, TriplesPrev),
				append(TriplesPrev, Triples, TriplesNext),
				nb_setval(semantics, TriplesNext)
			;	tr_n3p(Triples, Src, Mode)
			),
			Tokens = []
		),
		Exc2,
		(	(	wcacher(Arg, File)
			->	format(user_error, '** ~w ** ~w FROM ~w ** ~w~n', [Ise, Arg, File, Exc2])
			;	format(user_error, '** ~w ** ~w ** ~w~n', [Ise, Arg, Exc2])
			),
			flush_output(user_error),
			ignore(Parsed = fail)
		)
	),
	ignore(Parsed = true),
	(	Mode = semantics
	->	nb_getval(semantics, TriplesFinal),
		clist(TriplesFinal, Graph),
		write(semantics(Src, Graph)),
		writeln('.')
	;	true
	),
	told,
	(	File = '-'
	->	true
	;	close(In)
	),
	(	retract(tmpfile(Tmp))
	->	delete_file(Tmp)
	;	true
	),
	(	call(Parsed)
	->	(	flag(n3p)
		->	forall(
				(	pfx(Pp, Pu),
					\+wpfx(Pp)
				),
				(	portray_clause(pfx(Pp, Pu)),
					assertz(wpfx(Pp))
				)
			)
		;	true
		),
		open(Tmp_p, read, Rs, [encoding(utf8)]),
		(	Mode = semantics
		->	repeat,
			read(Rs, Rt),
			(	Rt = end_of_file
			->	true
			;	djitis(Rt),
				(	Rt = semantics(_, cn(L))
				->	length(L, N),
					nb_setval(sc, N)
				;	Rt \= semantics(_, true),
					nb_setval(sc, 1)
				),
				fail
			)
		;	repeat,
			read(Rs, Rt),
			(	Rt = end_of_file
			->	true
			;	(	predicate_property(Rt, dynamic),
					functor(Rt, P, 2)
				->	(	\+pred(P),
						P \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>',
						P \= query
					->	assertz(pred(P)),
						(	flag(n3p)
						->	portray_clause(pred(P))
						;	true
						)
					;	true
					)
				;	true
				),
				(	ground(Rt),
					Rt \= ':-'(_, _)
				->	(	Rt = dynapred(B/2)
					->	(	flag(n3p)
						->	format(':- dynamic(~q).~n', [B/2]),
							format(':- multifile(~q).~n', [B/2])
						;	true
						)
					;	(	predicate_property(Rt, dynamic)
						->	true
						;	close(Rs),
							(	retract(tmpfile(Tmp_p))
							->	delete_file(Tmp_p)
							;	true
							),
							throw(builtin_redefinition(Rt))
						),
						(	Rt \= implies(_, _, _),
							\+flag('no-distinct-input'),
							call(Rt)
						->	true
						;	djitis(Rt),
							cnt(sc),
							(	flag(n3p)
							->	portray_clause(Rt)
							;	true
							)
						)
					)
				;	(	Rt = prfstep(Ct, _, Pt, _, Qt, It, Mt, St)
					->	term_index(Ct, Cnd),
						term_index(Pt, Pnd),
						(	nonvar(It)
						->	copy_term_nat(It, Ic)
						;	Ic = It
						),
						(	\+prfstep(Ct, Cnd, Pt, Pnd, Qt, Ic, Mt, St)
						->	assertz(prfstep(Ct, Cnd, Pt, Pnd, Qt, Ic, Mt, St))
						;	true
						)
					;	(	Rt = ':-'(Ci, Pi)
						->	(	Ci = true
							->	(	flag(n3p)
								->	cn_conj(Pi, Pc),
									portray_clause(':-'(Pc))
								;	call(Pi)
								)
							;	atomic_list_concat(['<', Arg, '>'], Si),
								copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(Pi, Ci), Ri),
								cn_conj(Pi, Pn),
								(	flag(nope)
								->	Ph = Pn
								;	(	Pi = when(Ai, Bi)
									->	c_append(Bi, istep(Si, Pi, Ci, Ri), Bh),
										Ph = when(Ai, Bh)
									;	c_append(Pn, istep(Si, Pi, Ci, Ri), Ph)
									)
								),
								(	flag('rule-histogram')
								->	(	Ph = when(Ak, Bk)
									->	c_append(Bk, pstep(Ri), Bj),
										Pj = when(Ak, Bj)
									;	c_append(Ph, pstep(Ri), Pj)
									)
								;	Pj = Ph
								),
								cnt(sc),
								functor(Ci, CPi, _),
								(	flag(n3p)
								->	portray_clause(cpred(CPi)),
									portray_clause(':-'(Ci, Pn))
								;	(	\+cpred(CPi)
									->	assertz(cpred(CPi))
									;	true
									),
									assertz(':-'(Ci, Pj))
								)
							)
						;	djitis(Rt),
							cnt(sc),
							(	flag(n3p)
							->	portray_clause(Rt)
							;	true
							)
						)
					)
				),
				fail
			)
		),
		close(Rs),
		(	retract(tmpfile(Tmp_p))
		->	delete_file(Tmp_p)
		;	true
		),
		nb_getval(sc, SC),
		nb_getval(input_statements, IN),
		Inp is SC+IN,
		nb_setval(input_statements, Inp),
		format(user_error, 'SC=~w~n', [SC]),
		flush_output(user_error)
	;	(	retract(tmpfile(Tmp_p))
		->	catch(delete_file(Tmp_p), _, true)
		;	true
		),
		(	\+flag('ignore-syntax-error'),
			\+flag('multi-query')
		->	nl,
			flush_output,
			halt(1)
		;	true
		)
	),
	!.


tr_n3p([], _, _) :-
	!.
% DEPRECATED
tr_n3p(['\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)|Z], Src, trules) :-
	!,
	(	clast(X, '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#true>\''(_, T))
	->	true
	;	T = 1.0
	),
	clist(L, X),
	tr_split(L, K, M),
	clist(K, N),
	write(implies(N, '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>\''([Y|M], T), Src)),
	writeln('.'),
	tr_n3p(Z, Src, trules).
% DEPRECATED
tr_n3p(['\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)|Z], Src, tquery) :-
	!,
	clist(U, X),
	tr_split(U, K, M),
	append(K, ['\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>\''([Y|M], T)], J),
	clist(J, N),
	write(implies(N, answer('\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>\'', [Y|M], T, _, _, _, _), Src)),
	writeln('.'),
	tr_n3p(Z, Src, tquery).
tr_n3p(['\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)|Z], Src, query) :-
	!,
	(	Y = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, T)
	->	(	is_list(T)
		->	H = T
		;	findvars(X, U, epsilon),
			distinct(U, H)
		),
		nb_setval(csv_header, H),
		V = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, H)
	;	V = Y
	),
	(	\+flag('limited-answer', _),
		flag(nope),
		(	flag('no-distinct-output')
		;	V = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, _)
		)
	->	write(query(X, V)),
		writeln('.')
	;	djiti(answer(V), A),
		write(implies(X, A, Src)),
		writeln('.')
	),
	tr_n3p(Z, Src, query).
tr_n3p([':-'(Y, X)|Z], Src, query) :-
	!,
	(	Y = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, T)
	->	(	is_list(T)
		->	H = T
		;	findvars(X, U, epsilon),
			distinct(U, H)
		),
		nb_setval(csv_header, H),
		V = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, H)
	;	V = Y
	),
	(	(	\+flag('limited-answer', _)
		;	V = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, _),
			flag(strings)
		),
		flag(nope)
	->	write(query(X, V)),
		writeln('.')
	;	djiti(answer(V), A),
		write(implies(X, A, Src)),
		writeln('.')
	),
	tr_n3p(Z, Src, query).
tr_n3p([X|Z], Src, query) :-
	!,
	(	\+flag('limited-answer', _),
		flag(nope),
		(	flag('no-distinct-output')
		;	X = '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>\''(_, _)
		)
	->	write(query(true, X)),
		writeln('.')
	;	djiti(answer(X), A),
		write(implies(true, A, Src)),
		writeln('.')
	),
	tr_n3p(Z, Src, query).
tr_n3p(['\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)|Z], Src, Mode) :-
	!,
	(	forall(
			(	cmember(I, X)
			),
			(	I = '\'<http://www.w3.org/2000/10/swap/log#implies>\''(J, _),
				findvars(J, K, beta),
				K \= []
			->	throw('premise_rule_may_not_contain_existential_in_premise'('\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)))
			;	true
			)
		)
	),
	(	Y = '\'<http://www.w3.org/2000/10/swap/log#implies>\''(U, _),
		findvars(U, V, beta),
		V \= []
	->	throw('derived_rule_may_not_contain_existential_in_premise'('\'<http://www.w3.org/2000/10/swap/log#implies>\''(X, Y)))
	;	true
	),
	(	flag(tactic, 'linear-select')
	->	write(implies(X, '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>\''(X, Y), Src)),
		writeln('.'),
		write(implies('\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>\''(X, Y), Y, Src)),
		writeln('.')
	;	write(implies(X, Y, Src)),
		writeln('.')
	),
	tr_n3p(Z, Src, Mode).
tr_n3p([':-'(Y, X)|Z], Src, Mode) :-
	!,
	findvars(Y, V, beta),
	(	V = []
	->	true
	;	throw('backward_rule_may_not_contain_existential_in_conclusion'(':-'(Y, X)))
	),
	write(':-'(Y, X)),
	writeln('.'),
	tr_n3p(Z, Src, Mode).
tr_n3p(['\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tactic>\''(X, Y)|Z], Src, Mode) :-
	!,
	write('\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tactic>\''(X, Y)),
	writeln('.'),
	tr_n3p(Z, Src, Mode).
tr_n3p([X|Z], Src, Mode) :-
	tr_tr(X, Y),
	(	findvars(Y, U, epsilon),
		U = []
	->	write(Y),
		writeln('.'),
		(	flag(nope)
		->	true
		;	write(prfstep(Y, _, true, _, Y, _, forward, Src)),
			writeln('.')
		)
	;	term_index(Y, A),
		write(':-'(Y, pass(A))),
		writeln('.')
	),
	tr_n3p(Z, Src, Mode).


tr_tr([], []) :-
	!.
tr_tr([A|B], [C|D]) :-
	!,
	tr_tr(A, C),
	tr_tr(B, D).
tr_tr(A, B) :-
	atom(A),
	!,
	(	atom_concat('_', C, A),
		(	atom_concat('bn_', _, C)
		;	atom_concat('e_', _, C)
		)
	->	nb_getval(var_ns, Vns),
		atomic_list_concat(['\'<', Vns, C, '>\''], B)
	;	B = A
	).
tr_tr(A, A) :-
	number(A),
	!.
tr_tr(A, B) :-
	A =.. [C|D],
	tr_tr(D, E),
	B =.. [C|E].


tr_split([], [], []) :-
	!.
tr_split([A|B], C, [A|D]) :-
	functor(A, '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>\'', _),
	!,
	tr_split(B, C, D).
tr_split([A|B], C, D) :-
	functor(A, '\'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#true>\'', _),
	!,
	tr_split(B, C, D).
tr_split([A|B], [A|C], D) :-
	tr_split(B, C, D).


% N3 parser

% according to http://www.w3.org/2000/10/swap/grammar/n3-ietf.txt
% inspired by http://code.google.com/p/km-rdf/wiki/Henry

barename(BareName, [name(BareName)|L2], L2).


barename_csl([BareName|Tail], L1, L3) :-
	barename(BareName, L1, L2),
	!,
	barename_csl_tail(Tail, L2, L3).
barename_csl([], L1, L1).


barename_csl_tail([BareName|Tail], [','|L2], L4) :-
	!,
	barename(BareName, L2, L3),
	barename_csl_tail(Tail, L3, L4).
barename_csl_tail([], L1, L1).


% DEPRECATED
boolean(true, [atname('true')|L2], L2) :-
	!.
boolean(true, [name('true')|L2], L2) :-
	!.
% DEPRECATED
boolean(false, [atname('false')|L2], L2) :-
	!.
boolean(false, [name('false')|L2], L2) :-
	!.
boolean(Boolean, L1, L2) :-
	literal(Atom, type(T), L1, L2),
	T = '\'<http://www.w3.org/2001/XMLSchema#boolean>\'',
	memberchk([Boolean, Atom], [[true, '\'true\''], [true, true], [true, '\'1\''], [false, '\'false\''], [false, false], [false, '\'0\'']]).


% DEPRECATED
declaration([atname(base)|L2], L3) :-
	!,
	explicituri(U, L2, L3),
	base_uri(V),
	resolve_uri(U, V, URI),
	retractall(base_uri(_)),
	assertz(base_uri(URI)).
declaration([name(Name)|L2], L4) :-
	downcase_atom(Name, 'base'),
	!,
	explicituri(U, L2, L3),
	base_uri(V),
	resolve_uri(U, V, URI),
	retractall(base_uri(_)),
	assertz(base_uri(URI)),
	withoutdot(L3, L4).
% DEPRECATED
declaration([atname(keywords)|L2], L3) :-
	!,
	barename_csl(List, L2, L3),
	retractall(keywords(_)),
	assertz(keywords(List)).
% DEPRECATED
declaration([atname(prefix)|L2], L4) :-
	!,
	prefix(Prefix, L2, L3),
	explicituri(U, L3, L4),
	base_uri(V),
	resolve_uri(U, V, URI),
	retractall(ns(Prefix, _)),
	assertz(ns(Prefix, URI)),
	put_pfx(Prefix, URI).
declaration([name(Name)|L2], L5) :-
	downcase_atom(Name, 'prefix'),
	prefix(Prefix, L2, L3),
	explicituri(U, L3, L4),
	base_uri(V),
	resolve_uri(U, V, URI),
	retractall(ns(Prefix, _)),
	assertz(ns(Prefix, URI)),
	put_pfx(Prefix, URI),
	withoutdot(L4, L5).


document(Triples, L1, L2) :-
	statements_optional(Triples, L1, L2).


dtlang(lang(Langcode), [atname(Name)|L2], L2) :-
	!,
	atomic_list_concat(['\'', Name, '\''], Langcode).
dtlang(type(Datatype), [caretcaret|L2], L3) :-
	!,
	uri(Datatype, L2, L3).
dtlang(type('\'<http://www.w3.org/2001/XMLSchema#string>\''), L1, L1).


% DEPRECATED
existential([atname(forSome)|L2], L3) :-
	!,
	symbol_csl(Symbols, L2, L3),
	nb_getval(fdepth, D),
	forall(
		(	member(S, Symbols)
		),
		(	gensym('qe_', Q),
			asserta(qevar(S, Q, D))
		)
	).


explicituri(ExplicitURI, [relative_uri(ExplicitURI)|L2], L2).


expression(Node, T, L1, L3) :-
	pathitem(N1, T1, L1, L2),
	pathtail(N1, Node, T2, L2, L3),
	append(T1, T2, T),
	(	keywords(List),
		memberchk(Node, List)
	->	nb_getval(line_number, Ln),
		throw(invalid_keyword_use(Node, after_line(Ln)))
	;	true
	).


formulacontent(Formula, L1, L2) :-
	statementlist(L, L1, L2),
	clist(L, Formula).


literal(Atom, DtLang, L1, L3) :-
	string(Codes, L1, L2),
	dtlang(DtLang, L2, L3),
	escape_string(Codes, B),
	escape_string(B, C),
	atom_codes(A, C),
	(	sub_atom(A, _, 1, _, '\'')
	->	escape_squote(C, D),
		atom_codes(E, D)
	;	E = A
	),
	atomic_list_concat(['\'', E, '\''], Atom).


numericliteral(Number, [numeric(Type, NumB)|L2], L2) :-
	numeral(NumB, NumC),
	(	flag(strict),
		Type = decimal
	->	rdiv_codes(Number, NumC)
	;	number_codes(Number, NumC)
	).


object(Node, Triples, L1, L2) :-
	expression(Node, Triples, L1, L2).


objecttail(Subject, Verb, [Triple|T], [','|L2], L4) :-
	!,
	object(Object, Triples, L2, L3),
	objecttail(Subject, Verb, Tail, L3, L4),
	append(Triples, Tail, T),
	(	Verb = isof(V)
	->	(	atom(V),
			\+sub_atom(V, 0, 1, _, '_')
		->	Triple =.. [V, Object, Subject]
		;	Triple = exopred(V, Object, Subject)
		)
	;	(	atom(Verb),
			\+sub_atom(Verb, 0, 1, _, '_')
		->	Triple =.. [Verb, Subject, Object]
		;	Triple = exopred(Verb, Subject, Object)
		)
	).
objecttail(_, _, [], L1, L1).


pathitem([], [], L1, L2) :-
	symbol(S, L1, L2),
	S = '\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>\'',
	!.
pathitem(Name, [], L1, L2) :-
	symbol(S, L1, L2),
	!,
	(	qevar(S, N, D),
		\+quvar(S, _, _)
	->	(	D >= 1,
			nb_getval(fdepth, FD),
			FD >= D,
			\+flag('pass-all-ground')
		->	atom_concat('_', N, Name),
			nb_setval(smod, false)
		;	nb_getval(var_ns, Vns),
			atomic_list_concat(['\'<', Vns, N, '>\''], Name)
		)
	;	(	quvar(S, N, D)
		->	(	(	D = 1,
					nb_getval(fdepth, FD),
					FD >= 1
				;	flag('pass-all-ground')
				)
			->	nb_getval(var_ns, Vns),
				atomic_list_concat(['\'<', Vns, N, '>\''], Name)
			;	atom_concat('_', N, Name),
				nb_setval(smod, false)
			)
		;	Name = S
		)
	),
	(	quvar(S, _, _)
	->	nb_setval(smod, false)
	;	true
	).
pathitem(VarID, [], [uvar(Var)|L2], L2) :-
	!,
	atom_codes(Var, VarCodes),
	subst([[[0'-], [0'_, 0'M, 0'I, 0'N, 0'U, 0'S, 0'_]], [[0'.], [0'_, 0'D, 0'O, 0'T, 0'_]]], VarCodes, VarTidy),
	atom_codes(VarAtom, [0'_|VarTidy]),
	(	flag('pass-all-ground')
	->	nb_getval(var_ns, Vns),
		atom_codes(VarFrag, VarTidy),
		atomic_list_concat(['\'<', Vns, VarFrag, '>\''], VarID)
	;	VarID = VarAtom
	),
	nb_setval(smod, false).
pathitem(Number, [], L1, L2) :-
	numericliteral(Number, L1, L2),
	!.
pathitem(Boolean, [], L1, L2) :-
	boolean(Boolean, L1, L2),
	!.
pathitem(Atom, [], L1, L2) :-
	literal(A, type(T), L1, L2),
	T = '\'<http://eulersharp.sourceforge.net/2003/03swap/prolog#atom>\'',
	!,
	atom_codes(A, B),
	escape_string(C, B),
	atom_codes(Atom, C).
pathitem(Number, [], L1, L2) :-
	literal(Atom, type(Type), L1, L2),
	memberchk(Type, ['\'<http://www.w3.org/2001/XMLSchema#integer>\'', '\'<http://www.w3.org/2001/XMLSchema#decimal>\'', '\'<http://www.w3.org/2001/XMLSchema#double>\'']),
	sub_atom(Atom, 1, _, 1, A),
	atom_codes(A, NumB),
	numeral(NumB, NumC),
	(	flag(strict),
		Type = '\'<http://www.w3.org/2001/XMLSchema#decimal>\''
	->	rdiv_codes(Number, NumC)
	;	number_codes(Number, NumC)
	),
	!.
pathitem(literal(Atom, DtLang), [], L1, L2) :-
	literal(Atom, DtLang, L1, L2),
	!.
pathitem(BNode, Triples, ['['|L2], L4) :-
	!,
	gensym('bn_', S),
	(	(	nb_getval(fdepth, 0)
		;	flag('pass-all-ground')
		)
	->	nb_getval(var_ns, Vns),
		atomic_list_concat(['\'<', Vns, S, '>\''], BN)
	;	atom_concat('_', S, BN),
		nb_setval(smod, false)
	),
	propertylist(BN, T, L2, [']'|L4]),
	(	memberchk('\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>\''(X, Head), T),
		memberchk('\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>\''(X, Tail), T),
		del(T, '\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\''(X, '\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#List>\''), U),
		del(U, '\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>\''(X, Head), V),
		del(V, '\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>\''(X, Tail), W)
	->	BNode = [Head|Tail],
		Triples = W
	;	BNode = BN,
		Triples = T
	).
pathitem(set(Distinct), Triples, ['(', '$'|L2], L4) :-
	!,
	pathlist(List, Triples, L2, ['$', ')'|L4]),
	(	nb_getval(smod, true)
	->	sort(List, Distinct)
	;	distinct(List, Distinct)
	).
pathitem(List, Triples, ['('|L2], L4) :-
	!,
	pathlist(List, Triples, L2, [')'|L4]).
pathitem(Node, [] , ['{'|L2], L4):-
	nb_getval(fdepth, I),
	J is I+1,
	nb_setval(fdepth, J),
	nb_setval(smod, true),
	formulacontent(Node, L2, ['}'|L4]),
	retractall(quvar(_, _, J)),
	retractall(qevar(_, _, J)),
	retractall(evar(_, _, J)),
	nb_setval(fdepth, I),
	nb_setval(smod, false).


pathlist([Node|Rest], Triples, L1, L3) :-
	expression(Node, T, L1, L2),
	!,
	pathlist(Rest, Tail, L2, L3),
	append(T, Tail, Triples).
pathlist([], [], L1, L1).


pathtail(Node, PNode, [Triple|Triples], ['!'|L2], L4) :-
	!,
	pathitem(Item, Triples2, L2, L3),
	prolog_verb(Item, Verb),
	dynamic_verb(Verb),
	gensym('bn_', S),
	(	(	nb_getval(fdepth, 0)
		;	flag('pass-all-ground')
		)
	->	nb_getval(var_ns, Vns),
		atomic_list_concat(['\'<', Vns, S, '>\''], BNode)
	;	atom_concat('_', S, BNode),
		nb_setval(smod, false)
	),
	(	Verb = isof(V)
	->	(	atom(V),
			\+sub_atom(V, 0, 1, _, '_')
		->	Triple =.. [V, BNode, Node]
		;	Triple = exopred(V, BNode, Node)
		)
	;	(	Verb = prolog:Pred
		->	(	BNode = true
			->	Triple =.. [Pred|Node]
			;	(	BNode = false
				->	T =.. [Pred|Node],
					Triple = \+(T)
				;	(	prolog_sym(_, Pred, func)
					->	T =.. [Pred|Node],
						Triple = is(BNode, T)
					;	Triple =.. [Pred, Node, BNode]
					)
				)
			)
		;	(	atom(Verb),
				\+sub_atom(Verb, 0, 1, _, '_')
			->	Triple =.. [Verb, Node, BNode]
			;	Triple = exopred(Verb, Node, BNode)
			)
		)
	),
	pathtail(BNode, PNode, Tail, L3, L4),
	append(Triples2, Tail, Triples).
pathtail(Node, PNode, [Triple|Triples], ['^'|L2], L4) :-
	!,
	pathitem(Item, Triples2, L2, L3),
	prolog_verb(Item, Verb),
	dynamic_verb(Verb),
	gensym('bn_', S),
	(	(	nb_getval(fdepth, 0)
		;	flag('pass-all-ground')
		)
	->	nb_getval(var_ns, Vns),
		atomic_list_concat(['\'<', Vns, S, '>\''], BNode)
	;	atom_concat('_', S, BNode),
		nb_setval(smod, false)
	),
	(	Verb = isof(V)
	->	(	atom(V),
			\+sub_atom(V, 0, 1, _, '_')
		->	Triple =.. [V, Node, BNode]
		;	Triple = exopred(V, Node, BNode)
		)
	;	(	Verb = prolog:Pred
		->	(	Node = true
			->	Triple =.. [Pred|BNode]
			;	(	Node = false
				->	T =.. [Pred|BNode],
					Triple = \+(T)
				;	(	prolog_sym(_, Pred, func)
					->	T =.. [Pred|BNode],
						Triple = is(Node, T)
					;	Triple =.. [Pred, BNode, Node]
					)
				)
			)
		;	(	atom(Verb),
				\+sub_atom(Verb, 0, 1, _, '_')
			->	Triple =.. [Verb, BNode, Node]
			;	Triple = exopred(Verb, BNode, Node)
			)
		)
	),
	pathtail(BNode, PNode, Tail, L3, L4),
	append(Triples2, Tail, Triples).
pathtail(Node, Node, [], L1, L1).


prefix(Prefix, [Prefix:''|L2], L2).


propertylist(Subject, [Triple|Triples], L1, L5) :-
	verb(Item, Triples1, L1, L2),
	prolog_verb(Item, Verb),
	dynamic_verb(Verb),
	!,
	object(Object, Triples2, L2, L3),
	objecttail(Subject, Verb, Triples3, L3, L4),
	propertylisttail(Subject, Triples4, L4, L5),
	append(Triples1, Triples2, Triples12),
	append(Triples12, Triples3, Triples123),
	append(Triples123, Triples4, Triples),
	(	Verb = isof(V)
	->	(	atom(V),
			\+sub_atom(V, 0, 1, _, '_')
		->	Triple =.. [V, Object, Subject]
		;	Triple = exopred(V, Object, Subject)
		)
	;	(	Verb = prolog:Pred
		->	(	Object = true
			->	Triple =.. [Pred|Subject]
			;	(	Object = false
				->	T =.. [Pred|Subject],
					Triple = \+(T)
				;	(	prolog_sym(_, Pred, func)
					->	T =.. [Pred|Subject],
						Triple = is(Object, T)
					;	Triple =.. [Pred, Subject, Object]
					)
				)
			)
		;	(	atom(Verb),
				\+sub_atom(Verb, 0, 1, _, '_')
			->	Triple =.. [Verb, Subject, Object]
			;	Triple = exopred(Verb, Subject, Object)
			)
		)
	).
propertylist(_, [], L1, L1).


propertylisttail(Subject, Triples, [';'|L2], L4) :-
	!,
	propertylisttailsemis(L2, L3),
	propertylist(Subject, Triples, L3, L4).
propertylisttail(_, [], L1, L1).


propertylisttailsemis([';'|L2], L3) :-
	!,
	propertylisttailsemis(L2, L3).
propertylisttailsemis(L1, L1).


qname(URI, [NS:Name|L2], L2) :-
	(	ns(NS, Base)
	->	atomic_list_concat([Base, Name], Name1),
		(	sub_atom(Name1, _, 1, _, '\'')
		->	atom_codes(Name1, Codes1),
			escape_squote(Codes1, Codes2),
			atom_codes(Name2, Codes2)
		;	Name2 = Name1
		),
		atomic_list_concat(['\'<', Name2, '>\''], URI)
	;	nb_getval(line_number, Ln),
		throw(no_prefix_directive(NS, after_line(Ln)))
	),
	!.


simpleStatement(Triples, L1, L3) :-
	subject(Subject, Triples1, L1, L2),
	(	Subject = (D1;D2)
	->	Triples = [(D1;D2)]
	;	propertylist(Subject, Triples2, L2, L3),
		append(Triples1, Triples2, Triples)
	).


statement([], L1, L2) :-
	declaration(L1, L2),
	!.
statement([], L1, L2) :-
	universal(L1, L2),
	!.
statement([], L1, L2) :-
	existential(L1, L2),
	!.
statement(Statement, L1, L2) :-
	simpleStatement(Statement, L1, L2).


statementlist(Triples, L1, L3) :-
	statement(Tr, L1, L2),
	!,
	statementtail(T, L2, L3),
	append(Tr, T, Triples).
statementlist([], L1, L1).


statements_optional(Triples, L1, L4) :-
	statement(Tr, L1, [dot(Ln)|L3]),
	!,
	nb_setval(line_number, Ln),
	statements_optional(T, L3, L4),
	append(Tr, T, Triples).
statements_optional([], L1, L1).


statementtail(T, [dot(Ln)|L2], L3) :-
	!,
	nb_setval(line_number, Ln),
	statementlist(T, L2, L3).
statementtail([], L1, L1).


string(Codes, [literal(Codes)|L2], L2).


subject(Node, Triples, L1, L2) :-
	expression(Node, Triples, L1, L2).


symbol(Name, L1, L2) :-
	uri(Name, L1, L2),
	!.
symbol(Name, [name(N)|L2], L2) :-
	!,
	(	keywords(List)
	->	(	memberchk(N, List)
		->	Name = N
		;	ns('', Base),
			atomic_list_concat(['\'<', Base, N, '>\''], Name)
		)
	;	(	memberchk(N, [true, false])
		->	Name = N
		;	nb_getval(line_number, Ln),
			throw(invalid_keyword(N, after_line(Ln)))
		)
	).
symbol(Name, [bnode(Label)|L2], L2) :-
	nb_getval(fdepth, D),
	(	D =:= 0
	->	N = Label
	;	atom_codes(Label, LabelCodes),
		subst([[[0'-], [0'_, 0'M, 0'I, 0'N, 0'U, 0'S, 0'_]], [[0'.], [0'_, 0'D, 0'O, 0'T, 0'_]]], LabelCodes, LabelTidy),
		atom_codes(N, LabelTidy)
	),
	(	evar(N, S, D)
	->	true
	;	atom_concat(N, '_', M),
		gensym(M, S),
		assertz(evar(N, S, D))
	),
	(	(	nb_getval(fdepth, 0)
		;	flag('pass-all-ground')
		)
	->	nb_getval(var_ns, Vns),
		(	flag('pass-all-ground')
		->	atomic_list_concat(['\'<', Vns, N, '>\''], Name)
		;	atomic_list_concat(['\'<', Vns, 'e_', S, '>\''], Name)
		)
	;	atom_concat('_e_', S, Name),
		nb_setval(smod, false)
	).


symbol_csl([Symbol|Tail], L1, L3) :-
	symbol(Symbol, L1, L2),
	!,
	symbol_csl_tail(Tail, L2, L3).
symbol_csl([], L1, L1).


symbol_csl_tail([Symbol|T], [','|L2], L4) :-
	!,
	symbol(Symbol, L2, L3),
	symbol_csl_tail(T, L3, L4).
symbol_csl_tail([], L1, L1).


% DEPRECATED
universal([atname(forAll)|L2], L3) :-
	!,
	symbol_csl(Symbols, L2, L3),
	nb_getval(fdepth, D),
	(	\+flag(traditional),
		D > 0
	->	throw(not_supported_keyword('@forAll', at_formula_depth(D)))
	;	true
	),
	forall(
		(	member(S, Symbols)
		),
		(	gensym('qu_', Q),
			asserta(quvar(S, Q, D))
		)
	).


uri(Name, L1, L2) :-
	explicituri(U, L1, L2),
	!,
	base_uri(V),
	resolve_uri(U, V, W),
	(	sub_atom(W, _, 1, _, '\'')
	->	atom_codes(W, X),
		escape_squote(X, Y),
		atom_codes(Z, Y)
	;	Z = W
	),
	atomic_list_concat(['\'<', Z, '>\''], Name).
uri(Name, L1, L2) :-
	qname(Name, L1, L2).


verb('\'<http://www.w3.org/2000/10/swap/log#implies>\'', [], ['=', '>'|L2], L2) :-
	!.
verb('\'<http://www.w3.org/2002/07/owl#sameAs>\'', [], ['='|L2], L2) :-
	!.
verb(':-', [], ['<', '='|L2], L2) :-
	!.
% DEPRECATED
verb('\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\'', [], [atname(a)|L2], L2) :-
	!.
verb('\'<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>\'', [], [name(a)|L2], L2) :-
	!.
% DEPRECATED
verb(Node, Triples, [atname(has)|L2], L3) :-
	!,
	expression(Node, Triples, L2, L3).
verb(Node, Triples, [name(has)|L2], L3) :-
	!,
	expression(Node, Triples, L2, L3).
% DEPRECATED
verb(isof(Node), Triples, [atname(is)|L2], L4) :-
	!,
	expression(Node, Triples, L2, [atname(of)|L4]).
verb(isof(Node), Triples, [name(is)|L2], L4) :-
	!,
	expression(Node, Triples, L2, [name(of)|L4]).
verb(Node, Triples, L1, L2) :-
	expression(Node, Triples, L1, L2).


withoutdot([dot(Ln)|L2], [dot(Ln)|L2]) :-
	!,
	throw(unexpected_dot(after_line(Ln))).
withoutdot(L1, [dot(Ln)|L1]) :-
	nb_getval(line_number, Ln).


% N3 tokenizer

tokens(In, List) :-
	get_code(In, C0),
	(	token(C0, In, C1, Tok1)
	->	true
	;	nb_getval(line_number, Ln),
		char_code(Char, C0),
		throw(illegal_token(char_code(Char, C0), line(Ln)))
	),
	(	Tok1 == end_of_file
	->	List = []
	;	List = [Tok1|Tokens],
		tokens(C1, In, Tokens)
	).


tokens(C0, In, List) :-
	(	token(C0, In, C1, H)
	->	true
	;	nb_getval(line_number, Ln),
		char_code(Char, C0),
		throw(illegal_token(char_code(Char, C0), line(Ln)))
	),
	(	H == end_of_file
	->	List = []
	;	List = [H|T],
		tokens(C1, In, T)
	).


token(-1, _, -1, end_of_file) :-
	!.
token(0'., In, C, Token) :-
	(	peek_code(In, C0),
		(	e(C0)
		->	T1 = [0'0|T2],
			get_code(In, CN1)
		;	0'0 =< C0,
			C0 =< 0'9,
			get_code(In, C1),
			integer_codes(C1, In, CN1, T1, T2)
		)
	->	(	exponent(CN1, In, C, T2)
		->	Type = double
		;	C = CN1,
			T2 = [],
			Type = decimal
		),
		Token = numeric(Type, [0'0, 0'.|T1])
	;	nb_getval(line_number, Ln),
		get_code(In, C),
		!,
		Token = dot(Ln)
	).
token(0'#, In, C, Token) :-
	!,
	get_code(In, C1),
	skip_line(C1, In, C2),
	token(C2, In, C, Token).
token(C0, In, C, Token) :-
	white_space(C0),
	!,
	get_code(In, C1),
	token(C1, In, C, Token).
token(C0, In, C, Number) :-
	0'0 =< C0,
	C0 =< 0'9,
	!,
	number_n(C0, In, C, Number).
token(0'-, In, C, Number) :-
	!,
	number_n(0'-, In, C, Number).
token(0'+, In, C, Number) :-
	!,
	number_n(0'+, In, C, Number).
token(0'", In, C, literal(Codes)) :-
	!,
	(	peek_code(In, 0'")
	->	get_code(In, 0'"),
		(	peek_code(In, 0'")
		->	get_code(In, 0'"),
			get_code(In, C1),
			dq_string(C1, In, C, Codes)
		;	get_code(In, C),
			Codes = []
		)
	;	get_code(In, C1),
		string_dq(C1, In, C, Codes)
	).
token(0'', In, C, literal(Codes)) :-
	!,
	(	peek_code(In, 0'')
	->	get_code(In, 0''),
		(	peek_code(In, 0'')
		->	get_code(In, 0''),
			get_code(In, C1),
			sq_string(C1, In, C, Codes)
		;	get_code(In, C),
			Codes = []
		)
	;	get_code(In, C1),
		string_sq(C1, In, C, Codes)
	).
token(0'?, In, C, uvar(Name)) :-
	!,
	get_code(In, C0),
	(	name(C0, In, C, Name)
	->	true
	;	C = C0,
		nb_getval(line_number, Ln),
		throw(empty_quickvar_name(line(Ln)))
	).
token(0'_, In, C, bnode(Name)) :-
	peek_code(In, 0':),
	!,
	get_code(In, _),
	get_code(In, C0),
	(	name(C0, In, C, Name)
	->	true
	;	C = C0,
		Name = ''
	).
token(0'<, In, C, relative_uri(URI)) :-
	peek_code(In, C1),
	C1 \== 0'=,
	!,
	get_code(In, C1),
	iri_chars(C1, In, C, Codes),
	D = Codes,
	atom_codes(URI, D).
token(0':, In, C, Token) :-
	!,
	get_code(In, C0),
	(	local_name(C0, In, C, Name)
	->	Token = '':Name
	;	Token = '':'',
		C = C0
	).
token(0'@, In, C, atname(Name)) :-
	get_code(In, C0),
	token(C0, In, C, name(Name)),
	!.
token(0'^, In, C, caretcaret) :-
	peek_code(In, 0'^),
	!,
	get_code(In, _),
	get_code(In, C).
token(C0, In, C, Token) :-
	name(C0, In, C1, Name),
	!,
	(	C1 == 0':
	->	get_code(In, C2),
		(	local_name(C2, In, C, Name2)
		->	Token = (Name:Name2)
		;	Token = (Name:''),
			C = C2
		)
	;	Token = name(Name),
		C = C1
	).
token(C0, In, C, P) :-
	punctuation(C0, P),
	!,
	get_code(In, C).


number_n(0'-, In, CN, numeric(T, [0'-|Codes])) :-
	!,
	get_code(In, C0),
	number_nn(C0, In, CN, numeric(T, Codes)).
number_n(0'+, In, CN, numeric(T, [0'+|Codes])) :-
	!,
	get_code(In, C0),
	number_nn(C0, In, CN, numeric(T, Codes)).
number_n(C0, In, CN, Value) :-
	number_nn(C0, In, CN, Value).


number_nn(C, In, CN, numeric(Type, Codes)) :-
	integer_codes(C, In, CN0, Codes, T0),
	(	CN0 == 0'.,
		peek_code(In, C0),
		(	e(C0)
		->	T1 = [0'0|T2],
			get_code(In, CN1)
		;	0'0 =< C0,
			C0 =< 0'9,
			get_code(In, C1),
			integer_codes(C1, In, CN1, T1, T2)
		),
		T0 = [0'.|T1]
	->	(	exponent(CN1, In, CN, T2)
		->	Type = double
		;	CN = CN1,
			T2 = [],
			Type = decimal
		)
	;	(	exponent(CN0, In, CN, T0)
		->	Type = double
		;	T0 = [],
			CN = CN0,
			Type = integer
		)
	).


integer_codes(C0, In, CN, [C0|T0], T) :-
	0'0 =< C0,
	C0 =< 0'9,
	!,
	get_code(In, C1),
	integer_codes(C1, In, CN, T0, T).
integer_codes(CN, _, CN, T, T).


exponent(C0, In, CN, [C0|T0]) :-
	e(C0),
	!,
	get_code(In, C1),
	optional_sign(C1, In, CN0, T0, T1),
	integer_codes(CN0, In, CN, T1, []),
	(	T1 = []
	->	nb_getval(line_number, Ln),
		throw(invalid_exponent(line(Ln)))
	;	true
	).


optional_sign(C0, In, CN, [C0|T], T) :-
	sign(C0),
	!,
	get_code(In, CN).
optional_sign(CN, _, CN, T, T).


e(0'e).
e(0'E).


sign(0'-).
sign(0'+).


dq_string(-1, _, _, []) :-
	!,
	nb_getval(line_number, Ln),
	throw(unexpected_end_of_input(line(Ln))).
dq_string(0'", In, C, []) :-
	(	retract(got_dq)
	->	true
	;	peek_code(In, 0'"),
		get_code(In, _)
	),
	(	retract(got_dq)
	->	assertz(got_dq)
	;	assertz(got_dq),
		peek_code(In, 0'"),
		get_code(In, _),
		assertz(got_dq)
	),
	!,
	(	peek_code(In, 0'")
	->	nb_getval(line_number, Ln),
		throw(unexpected_double_quote(line(Ln)))
	;	true
	),
	retractall(got_dq),
	get_code(In, C).
dq_string(0'", In, C, [0'"|T]) :-
	!,
	(	retract(got_dq)
	->	C1 = 0'"
	;	get_code(In, C1)
	),
	dq_string(C1, In, C, T).
dq_string(0'\\, In, C, [H|T]) :-
	(	retract(got_dq)
	->	C1 = 0'"
	;	get_code(In, C1)
	),
	!,
	string_escape(C1, In, C2, H),
	dq_string(C2, In, C, T).
dq_string(C0, In, C, [C0|T]) :-
	(	retract(got_dq)
	->	C1 = 0'"
	;	get_code(In, C1)
	),
	dq_string(C1, In, C, T).


sq_string(-1, _, _, []) :-
	!,
	nb_getval(line_number, Ln),
	throw(unexpected_end_of_input(line(Ln))).
sq_string(0'', In, C, []) :-
	(	retract(got_sq)
	->	true
	;	peek_code(In, 0''),
		get_code(In, _)
	),
	(	retract(got_sq)
	->	assertz(got_sq)
	;	assertz(got_sq),
		peek_code(In, 0''),
		get_code(In, _),
		assertz(got_sq)
	),
	!,
	(	peek_code(In, 0'')
	->	nb_getval(line_number, Ln),
		throw(unexpected_single_quote(line(Ln)))
	;	true
	),
	retractall(got_sq),
	get_code(In, C).
sq_string(0'', In, C, [0''|T]) :-
	!,
	(	retract(got_sq)
	->	C1 = 0''
	;	get_code(In, C1)
	),
	sq_string(C1, In, C, T).
sq_string(0'\\, In, C, [H|T]) :-
	(	retract(got_sq)
	->	C1 = 0''
	;	get_code(In, C1)
	),
	!,
	string_escape(C1, In, C2, H),
	sq_string(C2, In, C, T).
sq_string(C0, In, C, [C0|T]) :-
	(	retract(got_sq)
	->	C1 = 0''
	;	get_code(In, C1)
	),
	sq_string(C1, In, C, T).


string_dq(-1, _, _, []) :-
	!,
	nb_getval(line_number, Ln),
	throw(unexpected_end_of_input(line(Ln))).
string_dq(0'\n, _, _, []) :-
	!,
	nb_getval(line_number, Ln),
	throw(unexpected_end_of_line(line(Ln))).
string_dq(0'", In, C, []) :-
	!,
	get_code(In, C).
string_dq(0'\\, In, C, D) :-
	get_code(In, C1),
	!,
	string_escape(C1, In, C2, H),
	(	current_prolog_flag(windows, true),
		H > 0xFFFF
	->	E is (H-0x10000)>>10+0xD800,
		F is (H-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [H|T]
	),
	string_dq(C2, In, C, T).
string_dq(C0, In, C, D) :-
	(	current_prolog_flag(windows, true),
		C0 > 0xFFFF
	->	E is (C0-0x10000)>>10+0xD800,
		F is (C0-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [C0|T]
	),
	get_code(In, C1),
	string_dq(C1, In, C, T).


string_sq(-1, _, _, []) :-
	!,
	nb_getval(line_number, Ln),
	throw(unexpected_end_of_input(line(Ln))).
string_sq(0'', In, C, []) :-
	!,
	get_code(In, C).
string_sq(0'\\, In, C, D) :-
	get_code(In, C1),
	!,
	string_escape(C1, In, C2, H),
	(	current_prolog_flag(windows, true),
		H > 0xFFFF
	->	E is (H-0x10000)>>10+0xD800,
		F is (H-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [H|T]
	),
	string_sq(C2, In, C, T).
string_sq(C0, In, C, D) :-
	(	current_prolog_flag(windows, true),
		C0 > 0xFFFF
	->	E is (C0-0x10000)>>10+0xD800,
		F is (C0-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [C0|T]
	),
	get_code(In, C1),
	string_sq(C1, In, C, T).


string_escape(0't, In, C, 0'\t) :-
	!,
	get_code(In, C).
string_escape(0'b, In, C, 0'\b) :-
	!,
	get_code(In, C).
string_escape(0'n, In, C, 0'\n) :-
	!,
	get_code(In, C).
string_escape(0'r, In, C, 0'\r) :-
	!,
	get_code(In, C).
string_escape(0'f, In, C, 0'\f) :-
	!,
	get_code(In, C).
string_escape(0'", In, C, 0'") :-
	!,
	get_code(In, C).
string_escape(0'', In, C, 0'') :-
	!,
	get_code(In, C).
string_escape(0'\\, In, C, 0'\\) :-
	!,
	get_code(In, C).
string_escape(0'u, In, C, Code) :-
	!,
	get_hhhh(In, A),
	(	0xD800 =< A,
		A =< 0xDBFF
	->	get_code(In, 0'\\),
		get_code(In, 0'u),
		get_hhhh(In, B),
		Code is 0x10000+(A-0xD800)*0x400+(B-0xDC00)
	;	Code is A
	),
	get_code(In, C).
string_escape(0'U, In, C, Code) :-
	!,
	get_hhhh(In, Code0),
	get_hhhh(In, Code1),
	Code is Code0 << 16 + Code1,
	get_code(In, C).
string_escape(C, _, _, _) :-
	nb_getval(line_number, Ln),
	atom_codes(A, [0'\\, C]),
	throw(illegal_string_escape_sequence(A, line(Ln))).


get_hhhh(In, Code) :-
	get_code(In, C1),
	code_type(C1, xdigit(D1)),
	get_code(In, C2),
	code_type(C2, xdigit(D2)),
	get_code(In, C3),
	code_type(C3, xdigit(D3)),
	get_code(In, C4),
	code_type(C4, xdigit(D4)),
	Code is D1<<12+D2<<8+D3<<4+D4.


language(C0, In, C, [C0|Codes]) :-
	code_type(C0, lower),
	get_code(In, C1),
	lwr_word(C1, In, C2, Codes, Tail),
	sub_langs(C2, In, C, Tail, []).


lwr_word(C0, In, C, [C0|T0], T) :-
	code_type(C0, lower),
	!,
	get_code(In, C1),
	lwr_word(C1, In, C, T0, T).
lwr_word(C, _, C, T, T).


sub_langs(0'-, In, C, [0'-, C1|Codes], T) :-
	get_code(In, C1),
	lwrdig(C1),
	!,
	get_code(In, C2),
	lwrdigs(C2, In, C3, Codes, Tail),
	sub_langs(C3, In, C, Tail, T).
sub_langs(C, _, C, T, T).


lwrdig(C) :-
	code_type(C, lower),
	!.
lwrdig(C) :-
	code_type(C, digit).


lwrdigs(C0, In, C, [C0|T0], T) :-
	lwrdig(C0),
	!,
	get_code(In, C1),
	lwr_word(C1, In, C, T0, T).
lwrdigs(C, _, C, T, T).


iri_chars(0'>, In, C, []) :-
	!,
	get_code(In, C).
iri_chars(0'\\, In, C, D) :-
	!,
	get_code(In, C1),
	iri_escape(C1, In, C2, H),
	\+non_iri_char(H),
	(	current_prolog_flag(windows, true),
		H > 0xFFFF
	->	E is (H-0x10000)>>10+0xD800,
		F is (H-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [H|T]
	),
	iri_chars(C2, In, C, T).
iri_chars(0'%, In, C, [0'%, C1, C2|T]) :-
	!,
	get_code(In, C1),
	code_type(C1, xdigit(_)),
	get_code(In, C2),
	code_type(C2, xdigit(_)),
	get_code(In, C3),
	iri_chars(C3, In, C, T).
iri_chars(-1, _, _, _) :-
	!,
	fail.
iri_chars(C0, In, C, D) :-
	\+non_iri_char(C0),
	(	current_prolog_flag(windows, true),
		C0 > 0xFFFF
	->	E is (C0-0x10000)>>10+0xD800,
		F is (C0-0x10000) mod 0x400+0xDC00,
		D = [E, F|T]
	;	D = [C0|T]
	),
	get_code(In, C1),
	iri_chars(C1, In, C, T).


iri_escape(0'u, In, C, Code) :-
	!,
	get_hhhh(In, A),
	(	0xD800 =< A,
		A =< 0xDBFF
	->	get_code(In, 0'\\),
		get_code(In, 0'u),
		get_hhhh(In, B),
		Code is 0x10000+(A-0xD800)*0x400+(B-0xDC00)
	;	Code is A
	),
	get_code(In, C).
iri_escape(0'U, In, C, Code) :-
	!,
	get_hhhh(In, Code0),
	get_hhhh(In, Code1),
	Code is Code0 << 16 + Code1,
	get_code(In, C).
iri_escape(C, _, _, _) :-
	nb_getval(line_number, Ln),
	atom_codes(A, [0'\\, C]),
	throw(illegal_iri_escape_sequence(A, line(Ln))).


non_iri_char(0x20).
non_iri_char(0'<).
non_iri_char(0'>).
non_iri_char(0'").
non_iri_char(0'{).
non_iri_char(0'}).
non_iri_char(0'|).
non_iri_char(0'^).
non_iri_char(0'`).
non_iri_char(0'\\).


name(C0, In, C, Atom) :-
	name_start_char(C0),
	get_code(In, C1),
	name_chars(C1, In, C, T),
	atom_codes(Atom, [C0|T]).


name_start_char(C) :-
	pn_chars_base(C),
	!.
name_start_char(0'_).
name_start_char(C) :-
	code_type(C, digit).


name_chars(0'., In, C, [0'.|T]) :-
	peek_code(In, C1),
	pn_chars(C1),
	!,
	get_code(In, C1),
	name_chars(C1, In, C, T).
name_chars(C0, In, C, [C0|T]) :-
	pn_chars(C0),
	!,
	get_code(In, C1),
	name_chars(C1, In, C, T).
name_chars(C, _, C, []).


pn_chars_base(C) :-
	code_type(C, alpha),
	!.
pn_chars_base(C) :-
	0xC0 =< C,
	C =< 0xD6,
	!.
pn_chars_base(C) :-
	0xD8 =< C,
	C =< 0xF6,
	!.
pn_chars_base(C) :-
	0xF8 =< C,
	C =< 0x2FF,
	!.
pn_chars_base(C) :-
	0x370 =< C,
	C =< 0x37D,
	!.
pn_chars_base(C) :-
	0x37F =< C,
	C =< 0x1FFF,
	!.
pn_chars_base(C) :-
	0x200C =< C,
	C =< 0x200D,
	!.
pn_chars_base(C) :-
	0x2070 =< C,
	C =< 0x218F,
	!.
pn_chars_base(C) :-
	0x2C00 =< C,
	C =< 0x2FEF,
	!.
pn_chars_base(C) :-
	0x3001 =< C,
	C =< 0xD7FF,
	!.
pn_chars_base(C) :-
	0xF900 =< C,
	C =< 0xFDCF,
	!.
pn_chars_base(C) :-
	0xFDF0 =< C,
	C =< 0xFFFD,
	!.
pn_chars_base(C) :-
	0x10000 =< C,
	C =< 0xEFFFF.


pn_chars(C) :-
	code_type(C, csym),
	!.
pn_chars(C) :-
	pn_chars_base(C),
	!.
pn_chars(0'-) :-
	!.
pn_chars(0xB7) :-
	!.
pn_chars(C) :-
	0x0300 =< C,
	C =< 0x036F,
	!.
pn_chars(C) :-
	0x203F =< C,
	C =< 0x2040.


local_name(0'\\, In, C, Atom) :-
	!,
	get_code(In, C0),
	reserved_char_escapes(C0),
	get_code(In, C1),
	local_name_chars(C1, In, C, T),
	atom_codes(Atom, [C0|T]).
local_name(0'%, In, C, Atom) :-
	!,
	get_code(In, C0),
	code_type(C0, xdigit(_)),
	get_code(In, C1),
	code_type(C1, xdigit(_)),
	get_code(In, C2),
	local_name_chars(C2, In, C, T),
	atom_codes(Atom, [0'%, C0, C1|T]).
local_name(C0, In, C, Atom) :-
	local_name_start_char(C0),
	get_code(In, C1),
	local_name_chars(C1, In, C, T),
	atom_codes(Atom, [C0|T]).


local_name_chars(0'\\, In, C, [C0|T]) :-
	!,
	get_code(In, C0),
	reserved_char_escapes(C0),
	get_code(In, C1),
	local_name_chars(C1, In, C, T).
local_name_chars(0'%, In, C, [0'%, C0, C1|T]) :-
	!,
	get_code(In, C0),
	code_type(C0, xdigit(_)),
	get_code(In, C1),
	code_type(C1, xdigit(_)),
	get_code(In, C2),
	local_name_chars(C2, In, C, T).
local_name_chars(0'., In, C, [0'.|T]) :-
	peek_code(In, C1),
	(	local_name_char(C1)
	;	C1 = 0'.
	),
	!,
	get_code(In, C1),
	local_name_chars(C1, In, C, T).
local_name_chars(C0, In, C, [C0|T]) :-
	local_name_char(C0),
	!,
	get_code(In, C1),
	local_name_chars(C1, In, C, T).
local_name_chars(C, _, C, []).


local_name_start_char(C) :-
	name_start_char(C),
	!.
local_name_start_char(0':).
local_name_start_char(0'%).
local_name_start_char(0'\\).


local_name_char(C) :-
	pn_chars(C),
	!.
local_name_char(0':).
local_name_char(0'%).
local_name_char(0'\\).


reserved_char_escapes(0'~).
reserved_char_escapes(0'.).
reserved_char_escapes(0'-).
reserved_char_escapes(0'!).
reserved_char_escapes(0'$).
reserved_char_escapes(0'&).
reserved_char_escapes(0'').
reserved_char_escapes(0'().
reserved_char_escapes(0')).
reserved_char_escapes(0'*).
reserved_char_escapes(0'+).
reserved_char_escapes(0',).
reserved_char_escapes(0';).
reserved_char_escapes(0'=).
reserved_char_escapes(0'/).
reserved_char_escapes(0'?).
reserved_char_escapes(0'#).
reserved_char_escapes(0'@).
reserved_char_escapes(0'%).
reserved_char_escapes(0'_).


punctuation(0'(, '(').
punctuation(0'), ')').
punctuation(0'[, '[').
punctuation(0'], ']').
punctuation(0',, ',').
punctuation(0':, ':').
punctuation(0';, ';').
punctuation(0'{, '{').
punctuation(0'}, '}').
punctuation(0'?, '?').
punctuation(0'!, '!').
punctuation(0'^, '^').
punctuation(0'=, '=').
punctuation(0'<, '<').
punctuation(0'>, '>').
punctuation(0'$, '$').


skip_line(-1, _, -1) :-
	!.
skip_line(0xA, In, C) :-
	!,
	cnt(line_number),
	get_code(In, C).
skip_line(0xD, In, C) :-
	!,
	get_code(In, C).
skip_line(_, In, C) :-
	get_code(In, C1),
	skip_line(C1, In, C).


white_space(0x9).
white_space(0xA) :-
	cnt(line_number).
white_space(0xD).
white_space(0x20).


% Reasoning output

w0([]) :-
	!.
w0(['--image', _|A]) :-
	!,
	w0(A).
w0([A|B]) :-
	(	\+sub_atom(A, 1, _, _, '"'),
		sub_atom(A, _, 1, _, ' '),
		\+sub_atom(A, _, _, 1, '"')
	->	format(' "~w"', [A])
	;	format(' ~w', [A])
	),
	w0(B).


w1([]) :-
	!.
w1([A|B]) :-
	(	\+sub_atom(A, 1, _, _, '"'),
		sub_atom(A, _, 1, _, ' '),
		\+sub_atom(A, _, _, 1, '"')
	->	format(' "~w"', [A])
	;	format(' ~w', [A])
	),
	w1(B).


wh :-
	(	keep_skolem(_)
	->	nb_getval(var_ns, Vns),
		put_pfx('var', Vns)
	;	true
	),
	(	flag('no-qnames')
	->	true
	;	nb_setval(wpfx, false),
		forall(
			(	pfx(A, B),
				\+wpfx(A)
			),
			(	(	\+flag(traditional)
				->	format('PREFIX ~w ~w~n', [A, B])
				;	format('@prefix ~w ~w.~n', [A, B])
				),
				assertz(wpfx(A)),
				nb_setval(wpfx, true)
			)
		),
		(	nb_getval(wpfx, true)
		->	nl
		;	true
		)
	).


w3 :-
	wh,
	nb_setval(fdepth, 0),
	nb_setval(pdepth, 0),
	nb_setval(cdepth, 0),
	flag(nope),
	!,
	(	query(Q, A),
		catch(call(Q), _, fail),
		(	\+ground(Q)
		->	clist(La, A),
			partconc(Q, La, Lp),
			Lp \= [],
			clist(Lp, Ap)
		;	Ap = A
		),
		relabel(Ap, B),
		indent,
		wt(B),
		ws(B),
		write('.'),
		nl,
		(	A = cn(L)
		->	length(L, I),
			cnt(output_statements, I)
		;	cnt(output_statements)
		),
		fail
	;	true
	),
	(	answer(B1, B2, B3, B4, B5, B6, B7),
		relabel([B1, B2, B3, B4, B5, B6, B7], [C1, C2, C3, C4, C5, C6, C7]),
		djiti(answer(C), answer(C1, C2, C3, C4, C5, C6, C7)),
		indent,
		wt(C),
		ws(C),
		write('.'),
		nl,
		cnt(output_statements),
		fail
	;	nl
	).
w3 :-
	(	prfstep(answer(_, _, _, _, _, _, _), _, _, _, _, _, _, _),
		!,
		nb_setval(empty_gives, false),
		indent,
		write('[] '),
		wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
		write(' '),
		wp('<http://www.w3.org/2000/10/swap/reason#Proof>'),
		write(', '),
		wp('<http://www.w3.org/2000/10/swap/reason#Conjunction>'),
		write(';'),
		indentation(2),
		nl,
		indent,
		(	prfstep(answer(B1, B2, B3, B4, B5, B6, B7), _, B, Pnd, Cn, R, _, A),
			R =.. [P, S, O1],
			djiti(answer(O), O1),
			Rule =.. [P, S, O],
			relabel([B1, B2, B3, B4, B5, B6, B7], [C1, C2, C3, C4, C5, C6, C7]),
			djiti(answer(C), Cn),
			nb_setval(empty_gives, C),
			\+got_wi(A, B, Pnd, C, Rule),
			assertz(got_wi(A, B, Pnd, C, Rule)),
			wp('<http://www.w3.org/2000/10/swap/reason#component>'),
			write(' '),
			wi(A, B, C, Rule),
			write(';'),
			nl,
			indent,
			fail
		;	retractall(got_wi(_, _, _, _, _))
		),
		wp('<http://www.w3.org/2000/10/swap/reason#gives>'),
		(	nb_getval(empty_gives, true)
		->	write(' true.')
		;	write(' {'),
			indentation(2),
			(	prfstep(answer(B1, B2, B3, B4, B5, B6, B7), _, _, _, _, _, _, _),
				relabel([B1, B2, B3, B4, B5, B6, B7], [C1, C2, C3, C4, C5, C6, C7]),
				djiti(answer(C), answer(C1, C2, C3, C4, C5, C6, C7)),
				nl,
				indent,
				getvars(C, D),
				(	C = '<http://www.w3.org/2000/10/swap/log#implies>'(_, _)
				->	Q = allv
				;	Q = some
				),
				(	\+flag(traditional)
				->	true
				;	wq(D, Q)
				),
				wt(C),
				ws(C),
				write('.'),
				cnt(output_statements),
				fail
			;	true
			),
			indentation(-2),
			nl,
			indent,
			write('}.')
		),
		indentation(-2),
		nl,
		nl
	;	true
	),
	(	nb_getval(lemma_count, Lco),
		nb_getval(lemma_cursor, Lcu),
		Lcu < Lco
	->	repeat,
		cnt(lemma_cursor),
		nb_getval(lemma_cursor, Cursor),
		lemma(Cursor, Ai, Bi, Ci, _, Di),
		indent,
		wj(Cursor, Ai, Bi, Ci, Di),
		nl,
		nl,
		nb_getval(lemma_count, Cnt),
		Cursor = Cnt,
		!
	;	true
	).


wi('<>', _, rule(_, _, A), _) :-	% wi(Source, Premise, Conclusion, Rule)
	!,
	write('[ '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Fact>'),
	write('; '),
	wp('<http://www.w3.org/2000/10/swap/reason#gives>'),
	write(' '),
	wg(A),
	write(']').
wi(A, B, C, Rule) :-
	term_index(B-C, Ind),
	(	lemma(Cnt, A, B, C, Ind, Rule)
	->	true
	;	cnt(lemma_count),
		nb_getval(lemma_count, Cnt),
		assertz(lemma(Cnt, A, B, C, Ind, Rule))
	),
	write('<#lemma'),
	write(Cnt),
	write('>').


wj(Cnt, A, true, C, Rule) :-	% wj(Count, Source, Premise, Conclusion, Rule)
	var(Rule),
	C \= '<http://www.w3.org/2000/10/swap/log#implies>'(_, _),
	!,
	write('<#lemma'),
	write(Cnt),
	write('> '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Extraction>'),
	write('; '),
	nl,
	indentation(2),
	indent,
	wp('<http://www.w3.org/2000/10/swap/reason#gives>'),
	(	C = true
	->	write(' true;')
	;	write(' {'),
		nl,
		indentation(2),
		indent,
		(	C = rule(PVars, EVars, Rule)
		->	(	\+flag(traditional)
			->	true
			;	wq(PVars, allv),
				wq(EVars, some)
			),
			wt(Rule)
		;	labelvars([A, C], 0, _, avar),
			getvars(C, D),
			(	\+flag(traditional)
			->	true
			;	wq(D, some)
			),
			wt(C)
		),
		ws(C),
		write('.'),
		nl,
		indentation(-2),
		indent,
		write('};')
	),
	nl,
	indent,
	wp('<http://www.w3.org/2000/10/swap/reason#because>'),
	write(' [ '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Parsing>'),
	write('; '),
	wp('<http://www.w3.org/2000/10/swap/reason#source>'),
	write(' '),
	wt(A),
	write('].'),
	indentation(-2).
wj(Cnt, A, B, C, Rule) :-
	write('<#lemma'),
	write(Cnt),
	write('> '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Inference>'),
	write('; '),
	nl,
	indentation(2),
	indent,
	wp('<http://www.w3.org/2000/10/swap/reason#gives>'),
	(	C = true
	->	write(' true;')
	;	write(' {'),
		nl,
		Rule = '<http://www.w3.org/2000/10/swap/log#implies>'(Prem, Conc),
		unifiable(Prem, B, Bs),
		(	unifiable(Conc, C, Cs)
		->	true
		;	Cs = []
		),
		append(Bs, Cs, Ds),
		sort(Ds, Bindings),
		term_variables(Prem, PVars),
		term_variables(Conc, CVars),
		nb_getval(wn, W),
		labelvars([A, B, C], W, N, some),
		nb_setval(wn, N),
		labelvars([Rule, PVars, CVars], 0, _, avar),
		findall(V,
			(	member(V, CVars),
				\+member(V, PVars)
			),
			EVars
		),
		getvars(C, D),
		(	C = '<http://www.w3.org/2000/10/swap/log#implies>'(_, _)
		->	Q = allv
		;	Q = some
		),
		indentation(2),
		indent,
		(	\+flag(traditional)
		->	true
		;	wq(D, Q)
		),
		wt(C),
		ws(C),
		write('.'),
		nl,
		indentation(-2),
		indent,
		write('}; ')
	),
	nl,
	indent,
	wp('<http://www.w3.org/2000/10/swap/reason#evidence>'),
	write(' ('),
	indentation(2),
	wr(B),
	indentation(-2),
	nl,
	indent,
	write(');'),
	retractall(got_wi(_, _, _, _, _)),
	nl,
	indent,
	(	\+flag(traditional)
	->	true
	;	wb(Bindings)
	),
	wp('<http://www.w3.org/2000/10/swap/reason#rule>'),
	write(' '),
	wi(A, true, rule(PVars, EVars, Rule), _),
	write('.'),
	indentation(-2).


wr(exopred(P, S, O)) :-
	atom(P),
	!,
	U =.. [P, S, O],
	wr(U).
wr(cn([X])) :-
	!,
	wr(X).
wr(cn([X|Y])) :-
	!,
	wr(X),
	(	Y = [Z]
	->	true
	;	Z = cn(Y)
	),
	wr(Z).
wr(Z) :-
	term_index(Z, Cnd),
	prfstep(Z, Cnd, Y, _, Q, Rule, _, X),
	!,
	nl,
	indent,
	wi(X, Y, Q, Rule).
wr(Y) :-
	nl,
	indent,
	write('[ '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Fact>'),
	write('; '),
	wp('<http://www.w3.org/2000/10/swap/reason#gives>'),
	write(' '),
	(	(	Y = true
		;	Y = pass(_)
		)
	->	wt(Y)
	;	write('{'),
		labelvars(Y, 0, _, avar),
		getvars(Y, Z),
		(	\+flag(traditional)
		->	true
		;	wq(Z, some)
		),
		wt(Y),
		write('}')
	),
	write(']').


wt(X) :-
	var(X),
	!,
	write('?'),
	write(X).
wt(rdiv(X, Y)) :-
	number_codes(Y, [0'1|Z]),
	lzero(Z, Z),
	!,
	(	Z = []
	->	F = '~d.0'
	;	length(Z, N),
		number_codes(X, U),
		(	length(U, N)
		->	F = '0.~d'
		;	atomic_list_concat(['~', N, 'd'], F)
		)
	),
	(	flag('no-numerals')
	->	write('"')
	;	true
	),
	format(F, [X]),
	(	flag('no-numerals')
	->	write('"^^'),
		wt('<http://www.w3.org/2001/XMLSchema#decimal>')
	;	true
	).
wt(rdiv(X, Y)) :-
	!,
	(	flag('no-numerals')
	->	write('"')
	;	true
	),
	format('~g', [rdiv(X, Y)]),
	(	flag('no-numerals')
	->	write('"^^'),
		wt('<http://www.w3.org/2001/XMLSchema#decimal>')
	;	true
	).
wt(X) :-
	number(X),
	!,
	(	flag('no-numerals')
	->	dtlit([U, V], X),
		dtlit([U, V], W),
		wt(W)
	;	(	flag(strict),
			float(X)
		->	format('~16e', [X])
		;	write(X)
		)
	).
wt((X, Y)) :-
	!,
	wt(X),
	ws(X),
	write('.'),
	(	flag(strings)
	->	write(' ')
	;	nl,
		indent
	),
	wt(Y).
wt(cn([X])) :-
	!,
	wt(X).
wt(cn([X|Y])) :-
	!,
	wt(X),
	ws(X),
	write('.'),
	(	flag(strings)
	->	write(' ')
	;	nl,
		indent
	),
	(	Y = [Z]
	->	true
	;	Z = cn(Y)
	),
	wt(Z).
wt(set(X)) :-
	!,
	write('($'),
	wl(X),
	write(' $)').
wt([]) :-
	!,
	write('()').
wt([X|Y]) :-
	!,
	(	\+last_tail([X|Y], [])
	->	write('[ '),
		wt('<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>'),
		write(' '),
		wg(X),
		write('; '),
		wt('<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>'),
		write(' '),
		wt(Y),
		write(']')
	;	write('('),
		wg(X),
		wl(Y),
		write(')')
	).
wt(X) :-
	functor(X, _, A),
	(	A = 0,
		!,
		wt0(X)
	;	A = 1,
		!,
		wt1(X)
	;	A = 2,
		!,
		wt2(X)
	;	wtn(X)
	).


wt0(!) :-
	!,
	write('() '),
	wp(!),
	write(' true').
wt0(fail) :-
	!,
	write('() '),
	wp(fail),
	write(' true').
wt0(X) :-
	atom(X),
	atom_concat(some, Y, X),
	!,
	(	\+flag('no-qvars'),
		\+flag('no-blank')	% DEPRECATED
	->	(	rule_uvar(L),
			(	ncllit
			->	(	memberchk(X, L)
				->	true
				;	retract(rule_uvar(L)),
					assertz(rule_uvar([X|L]))
				)
			;	memberchk(X, L)
			)
		->	write('?U_')
		;	write('_:sk_')
		),
		write(Y)
	;	nb_getval(var_ns, Vns),
		atomic_list_concat(['<', Vns, 'sk_', Y, '>'], Z),
		wt0(Z)
	).
wt0(X) :-
	atom(X),
	atom_concat(allv, Y, X),
	!,
	(	\+flag('no-qvars'),
		\+flag('pass-all-ground')
	->	(	rule_uvar(L),
			(	ncllit
			->	(	memberchk(X, L)
				->	true
				;	retract(rule_uvar(L)),
					assertz(rule_uvar([X|L]))
				)
			;	memberchk(X, L)
			)
		->	write('?U_')
		;	write('_:sk_')
		),
		write(Y)
	;	nb_getval(var_ns, Vns),
		atomic_list_concat(['<', Vns, 'U_', Y, '>'], Z),
		wt0(Z)
	).
wt0(X) :-
	atom(X),
	atom_concat(avar, Y, X),
	!,
	nb_getval(var_ns, Vns),
	atomic_list_concat(['<', Vns, 'x_', Y, '>'], Z),
	wt0(Z).
wt0(X) :-
	(	\+flag(traditional)
	->	true
	;	flag(nope)
	),
	\+flag('pass-all-ground'),
	\+keep_skolem(X),
	nb_getval(var_ns, Vns),
	sub_atom(X, 1, I, _, Vns),
	J is I+1,
	sub_atom(X, J, _, 1, Y),
	\+sub_atom(Y, 0, 3, _, 'qe_'),
	(	getlist(X, M)
	->	wt(M)
	;	(	rule_uvar(L),
			(	ncllit
			->	(	memberchk(Y, L)
				->	true
				;	retract(rule_uvar(L)),
					assertz(rule_uvar([Y|L]))
				)
			;	memberchk(Y, L)
			)
		->	(	\+flag(nope),
				sub_atom(Y, 0, 2, _, 'e_')
			->	write('_:')
			;	sub_atom(Y, 0, 2, _, Z),
				memberchk(Z, ['x_', 't_']),
				write('?')
			)
		;	(	\+flag('no-qvars'),
				\+flag('no-blank')	% DEPRECATED
			->	true
			;	flag('no-skolem', Prefix),
				sub_atom(X, 1, _, _, Prefix)
			),
			write('_:')
		),
		write(Y),
		(	sub_atom(Y, 0, 2, _, 'x_')
		->	write('_'),
			nb_getval(rn, N),
			write(N)
		;	true
		)
	),
	!.
wt0(X) :-
	flag('no-skolem', Prefix),
	(	\+flag(traditional)
	->	true
	;	flag(nope)
	),
	sub_atom(X, 1, _, _, Prefix),
	!,
	(	getlist(X, M)
	->	wt(M)
	;	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tuple>'(Y, ['no-skolem', Prefix, X]),
		wt0(Y)
	).
wt0(X) :-
	(	wtcache(X, W)
	->	true
	;	(	\+flag('no-qnames'),
			atom(X),
			(	sub_atom(X, I, 1, J, '#')
			->	J > 1,
				sub_atom(X, 0, I, _, C),
				atom_concat(C, '#>', D)
			;	J = 1,
				D = X
			),
			pfx(E, D),
			K is J-1,
			sub_atom(X, _, K, 1, F),
			atom_codes(F, G),
			atom_codes('^[A-Z_a-z][\\\\-0-9A-Z_a-z]*$', H),
			regex(H, G, _)
		->	atom_concat(E, F, W)
		;	(	\+flag(strings),
				atom(X),
				\+ (sub_atom(X, 0, 1, _, '<'), sub_atom(X, _, 1, 0, '>')),
				X \= true,
				X \= false
			->	W = literal(X, type('<http://eulersharp.sourceforge.net/2003/03swap/prolog#atom>'))
			;	W = X
			)
		),
		assertz(wtcache(X, W))
	),
	(	W = literal(X, type('<http://eulersharp.sourceforge.net/2003/03swap/prolog#atom>'))
	->	wt2(W)
	;	(	current_prolog_flag(windows, true)
		->	atom_codes(W, U),
			escape_unicode(U, V),
			atom_codes(Z, V)
		;	Z = W
		),
		write(Z)
	).


wt1(pass(_)) :-
	!,
	write('true').
wt1(X) :-
	X =.. [B|C],
	wt(C),
	write(' '),
	wp(B),
	write(' true').


wt2(literal(X, lang(Y))) :-
	!,
	write('"'),
	(	current_prolog_flag(windows, true)
	->	atom_codes(X, U),
		escape_unicode(U, V),
		atom_codes(Z, V)
	;	Z = X
	),
	write(Z),
	write('"@'),
	write(Y).
wt2(literal(X, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	!,
	write('"'),
	(	current_prolog_flag(windows, true)
	->	atom_codes(X, U),
		escape_unicode(U, V),
		atom_codes(Z, V)
	;	Z = X
	),
	write(Z),
	write('"').
wt2(literal(X, type(Y))) :-
	!,
	write('"'),
	(	current_prolog_flag(windows, true)
	->	atom_codes(X, U),
		escape_unicode(U, V),
		atom_codes(Z, V)
	;	Z = X
	),
	write(Z),
	write('"^^'),
	wt(Y).
wt2('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>'([X|Y], Z)) :-
	flag(nope),
	flag(tquery),	% DEPRECATED
	!,
	'<http://www.w3.org/2000/10/swap/log#conjunction>'(Y, U),
	write('{'),
	wt(U),
	write('. _: '),
	wp('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#true>'),
	write(' '),
	wt(Z),
	write('} '),
	wp('<http://www.w3.org/2000/10/swap/log#implies>'),
	write(' {'),
	wt(X),
	write('}').
wt2('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>'([X|Y], Z)) :-
	flag(nope),
	flag(tquery),	% DEPRECATED
	!,
	'<http://www.w3.org/2000/10/swap/log#conjunction>'(Y, U),
	write('{'),
	wt(U),
	write('. _: '),
	wp('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#true>'),
	write(' '),
	wt(Z),
	write('} '),
	wp('<http://www.w3.org/2000/10/swap/log#implies>'),
	write(' {'),
	wt(X),
	write('}').
wt2('<http://www.w3.org/2000/10/swap/log#implies>'(X, Y)) :-
	(	flag(nope)
	->	U = X
	;	(	X = when(A, B)
		->	c_append(B, istep(_, _, _, _), C),
			U = when(A, C)
		;	cn_conj(X, V),
			c_append(V, istep(_, _, _, _), U)
		)
	),
	(	flag('rule-histogram')
	->	(	U = when(D, E)
		->	c_append(E, pstep(_), F),
			Z = when(D, F)
		;	cn_conj(U, W),
			c_append(W, pstep(_), Z)
		)
	;	Z = U
	),
	(	rule_uvar(R)
	->	true
	;	R = [],
		cnt(rn)
	),
	(	nb_getval(pdepth, 0),
		nb_getval(cdepth, 0)
	->	assertz(rule_uvar(R))
	;	true
	),
	(	catch(clause(Y, Z), _, fail)
	->	(	nb_getval(fdepth, 0)
		->	assertz(ncllit)
		;	true
		),
		wg(Y),
		write(' <= '),
		wg(X),
		(	nb_getval(fdepth, 0)
		->	retract(ncllit)
		;	true
		)
	;	(	clause('<http://www.w3.org/2000/10/swap/log#implies>'(X, Y, _, _, _, _), true)
		->	wg(X),
			write(' => '),
			wg(Y)
		;	(	nb_getval(fdepth, 0)
			->	assertz(ncllit)
			;	true
			),
			(	\+atom(X)
			->	nb_getval(pdepth, PD),
				PD1 is PD+1,
				nb_setval(pdepth, PD1)
			;	true
			),
			wg(X),
			(	\+atom(X)
			->	nb_setval(pdepth, PD)
			;	true
			),
			(	nb_getval(fdepth, 0)
			->	retract(ncllit)
			;	true
			),
			write(' => '),
			(	\+atom(Y)
			->	nb_getval(cdepth, CD),
				CD1 is CD+1,
				nb_setval(cdepth, CD1)
			;	true
			),
			wg(Y),
			(	\+atom(Y)
			->	nb_setval(cdepth, CD)
			;	true
			)
		)
	),
	(	nb_getval(pdepth, 0),
		nb_getval(cdepth, 0)
	->	retract(rule_uvar(_))
	;	true
	),
	!.
wt2(':-'(X, Y)) :-
	(	rule_uvar(R)
	->	true
	;	R = [],
		cnt(rn)
	),
	(	nb_getval(fdepth, 0)
	->	assertz(ncllit)
	;	true
	),
	assertz(rule_uvar(R)),
	(	Y = true
	->	wt(X)
	;	wg(X),
		write(' <= '),
		wg(Y),
		retract(rule_uvar(_))
	),
	(	nb_getval(fdepth, 0)
	->	retract(ncllit)
	;	true
	),
	!.
wt2(is(O, T)) :-
	!,
	(	number(T),
		T < 0
	->	P = -,
		Q is -T,
		S = [Q]
	;	T =.. [P|S]
	),
	wg(S),
	write(' '),
	wp(P),
	write(' '),
	wg(O).
wt2(prolog:X) :-
	!,
	(	X = '\';\''
	->	Y = disjunction
	;	prolog_sym(Y, X, _)
	),
	atomic_list_concat(['<http://eulersharp.sourceforge.net/2003/03swap/prolog#', Y, '>'], Z),
	wt0(Z).
wt2(X) :-
	X =.. [P, S, O],
	(	P \= true,
		prolog_sym(_, P, _)
	->	wt([S, O]),
		write(' '),
		wp(P),
		write(' true')
	;	wg(S),
		write(' '),
		wp(P),
		write(' '),
		wg(O)
	).


wtn(exopred(P, S, O)) :-
	!,
	(	atom(P)
	->	X =.. [P, S, O],
		wt2(X)
	;	wg(S),
		write(' '),
		wg(P),
		write(' '),
		wg(O)
	).
wtn(X) :-
	X =.. [B|C],
	(	atom(B),
		\+sub_atom(B, 0, 1, _, '<'),
		\+prolog_sym(_, B, _),
		X \= true,
		X \= false
	->	wt([B|C]),
		write('^'),
		wp('<http://eulersharp.sourceforge.net/2003/03swap/prolog#univ>')
	;	wt(C),
		write(' '),
		wp(B),
		write(' true')
	).


wg(X) :-
	var(X),
	!,
	write('?'),
	write(X).
wg(X) :-
	functor(X, F, A),
	(	(	F = exopred,
			!
		;	F = cn,
			!
		;	prolog_sym(_, F, _),
			F \= true,
			F \= false,
			F \= '-',
			F \= /,
			!
		;	A >= 2,
			F \= '.',
			F \= '[|]',
			F \= ':',
			F \= literal,
			F \= rdiv
		)
	->	write('{'),
		indentation(2),
		nb_getval(fdepth, D),
		E is D+1,
		nb_setval(fdepth, E),
		wt(X),
		nb_setval(fdepth, D),
		indentation(-2),
		write('}')
	;	wt(X)
	).


wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>') :-
	\+flag('no-qnames'),
	!,
	write('a').
wp('<http://www.w3.org/2000/10/swap/log#implies>') :-
	\+flag('no-qnames'),
	!,
	write('=>').
wp(':-') :-
	\+flag('no-qnames'),
	!,
	write('<=').
wp(X) :-
	(	prolog_sym(Y, X, _),
		X \= true,
		X \= false
	->	atomic_list_concat(['<http://eulersharp.sourceforge.net/2003/03swap/prolog#', Y, '>'], Z),
		wt(Z)
	;	wg(X)
	).


wk([]) :-
	!.
wk([X|Y]) :-
	write(', '),
	wt(X),
	wk(Y).


wl([]) :-
	!.
wl([X|Y]) :-
	write(' '),
	wg(X),
	wl(Y).


wq([], _) :-
	!.
wq([X|Y], allv) :-
	!,
	write('@forAll '),
	wt(X),
	wk(Y),
	write('. ').
wq([X|Y], some) :-
	(	\+flag('no-qvars'),
		\+flag('no-blank')	% DEPRECATED
	->	write('@forSome '),
		wt(X),
		wk(Y),
		write('. ')
	;	true
	).


wb([]) :-
	!.
wb([X = Y|Z]) :-
	wp('<http://www.w3.org/2000/10/swap/reason#binding>'),
	write(' [ '),
	wp('<http://www.w3.org/2000/10/swap/reason#variable>'),
	write(' '),
	wv(X),
	write('; '),
	wp('<http://www.w3.org/2000/10/swap/reason#boundTo>'),
	write(' '),
	wv(Y),
	write('];'),
	nl,
	indent,
	wb(Z).


wv(X) :-
	atom(X),
	atom_concat(avar, Y, X),
	!,
	write('[ '),
	wp('<http://www.w3.org/2004/06/rei#uri>'),
	write(' "'),
	nb_getval(var_ns, Vns),
	write(Vns),
	write('x_'),
	write(Y),
	write('"]').
wv(X) :-
	atom(X),
	atom_concat(some, Y, X),
	!,
	write('[ '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Existential>'),
	write('; '),
	wp('<http://www.w3.org/2004/06/rei#nodeId>'),
	write(' "_:sk_'),
	write(Y),
	write('"]').
wv(X) :-
	atom(X),
	nb_getval(var_ns, Vns),
	sub_atom(X, 1, I, _, Vns),
	!,
	write('[ '),
	wp('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'),
	write(' '),
	wp('<http://www.w3.org/2000/10/swap/reason#Existential>'),
	write('; '),
	wp('<http://www.w3.org/2004/06/rei#nodeId>'),
	write(' "'),
	write(Vns),
	J is I+1,
	sub_atom(X, J, _, 1, Q),
	write(Q),
	write('"]').
wv(X) :-
	atom(X),
	sub_atom(X, 1, _, 1, U),
	atomic_list_concat(['<', U, '>'], X),
	!,
	write('[ '),
	wp('<http://www.w3.org/2004/06/rei#uri>'),
	write(' "'),
	write(U),
	write('"]').
wv(X) :-
	wg(X).


ws(cn(X)) :-
	!,
	last(X, Y),
	ws(Y).
ws(X) :-
	X =.. Y,
	(	flag(tquery)	% DEPRECATED
	->	true
	;	last(Y, Z),
		(	\+number(Z),
			Z \= rdiv(_, _)
		->	true
		;	write(' ')
		)
	).


wst :-
	findall([Key, Str],
		(	'<http://www.w3.org/2000/10/swap/log#outputString>'(Key, Str)
		;	answer(A1, A2, A3, A4, A5, A6, A7),
			djiti(answer('<http://www.w3.org/2000/10/swap/log#outputString>'(Key, Str)), answer(A1, A2, A3, A4, A5, A6, A7))
		),
		KS
	),
	sort(KS, KT),
	forall(
		(	member([_, MT], KT),
			getcodes(MT, LT)
		),
		(	escape_string(NT, LT),
			atom_codes(ST, NT),
			wt(ST)
		)
	),
	(	catch(nb_getval(csv_header, Header), _, Header = []),
		wct(Header, Header),
		length(Header, Headerl),
		query(Where, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>'(_, Select)),
		catch(call(Where), _, fail),
		write('\r\n'),
		wct(Select, Header),
		cnt(output_statements, Headerl),
		cnt(answer_count),
		nb_getval(answer_count, AnswerCount),
		(	flag('limited-answer', AnswerLimit),
			AnswerCount >= AnswerLimit
		->	true
		;	fail
		)
	;	true
	).


wct([], []) :-
	!.
wct([A], [C]) :-
	!,
	wcf(A, C).
wct([A|B], [C|D]) :-
	wcf(A, C),
	write(','),
	wct(B, D).


wcf(A, _) :-
	var(A),
	!.
wcf(rdiv(X, Y), _) :-
	number_codes(Y, [0'1|Z]),
	lzero(Z, Z),
	!,
	(	Z = []
	->	F = '~d.0'
	;	length(Z, N),
		number_codes(X, U),
		(	length(U, N)
		->	F = '0.~d'
		;	atomic_list_concat(['~', N, 'd'], F)
		)
	),
	format(F, [X]).
wcf(literal(A, B), _) :-
	!,
	atom_codes(A, C),
	subst([[[0'\\, 0'"], [0'", 0'"]]], C, E),
	atom_codes(F, E),
	(	B \= type('<http://www.w3.org/2001/XMLSchema#dateTime>'),
		B \= type('<http://www.w3.org/2001/XMLSchema#date>'),
		B \= type('<http://www.w3.org/2001/XMLSchema#time>'),
		B \= type('<http://www.w3.org/2001/XMLSchema#duration>'),
		B \= type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'),
		B \= type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')
	->	write('"'),
		write(F),
		write('"')
	;	write(F)
	).
wcf(A, _) :-
	atom(A),
	nb_getval(var_ns, Vns),
	sub_atom(A, 1, I, _, Vns),
	!,
	J is I+1,
	sub_atom(A, J, _, 1, B),
	write('_:'),
	write(B).
wcf(A, _) :-
	atom(A),
	flag('no-skolem', Prefix),
	sub_atom(A, 1, _, _, Prefix),
	!,
	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tuple>'(B, ['no-skolem', Prefix, A]),
	wt0(B).
wcf(A, B) :-
	atom(A),
	relabel(A, C),
	sub_atom(C, 0, 1, _, '<'),
	sub_atom(C, _, 1, 0, '>'),
	!,
	sub_atom(C, 1, _, 1, D),
	(	sub_atom(B, _, 2, 0, 'ID')
	->	(	flag('hmac-key', Key)
		->	hmac_sha(Key, D, E, [algorithm(sha1)])
		;	sha_hash(D, E, [algorithm(sha1)])
		),
		atom_codes(F, E),
		base64xml(F, G),
		write(G)
	;	write(D)
	).
wcf(A, _) :-
	atom(A),
	sub_atom(A, 0, 1, _, '_'),
	!,
	sub_atom(A, 1, _, 0, B),
	write(B).
wcf(A, _) :-
	with_output_to(atom(B), wg(A)),
	write(B).


indent:-
	nb_getval(indentation, A),
	tab(A).


indentation(C) :-
	nb_getval(indentation, A),
	B is A+C,
	nb_setval(indentation, B).



% ------------------------------------------------------
% EAM (Euler Abstract Machine) supporting Unifying Logic
% ------------------------------------------------------

% In a nutshell:
%
%  1/ Select rule P => C
%  2/ Prove P & NOT(C) (backward chaining) and if it fails backtrack to 1/
%  3/ If P & NOT(C) assert C (forward chaining) and remove brake
%  4/ If C = answer(A) and tactic limited-answer stop, else backtrack to 2/
%  5/ If brake or tactic linear-select stop, else start again at 1/

eam(Span) :-
	(	cnt(tr),
		(	(	flag(brake, BrakeLim)	% DEPRECATED
			;	flag('limited-brake', BrakeLim)
			),
			nb_getval(tr, TR),
			TR >= BrakeLim
		->	(	flag(strings)
			->	true
			;	w3
			),
			(	flag(brake, _)	% DEPRECATED
			->	throw(maximimum_brake_count(TR))
			;	throw(halt)
			)
		;	true
		),
		(	flag(debug)
		->	format(user_error, 'eam/1 entering span ~w~n', [Span]),
			flush_output(user_error)
		;	true
		),
		implies(Prem, Conc, Src),
		ignore(Prem = exopred(_, _, _)),
		(	flag(nope),
			\+flag('rule-histogram')
		->	true
		;	copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(Prem, Conc), Rule)
		),
		(	flag(debug)
		->	format(user_error, '. eam/1 selecting rule ~q~n', [implies(Prem, Conc, Src)]),
			flush_output(user_error)
		;	true
		),
		catch(call_residue_vars(Prem, Res), Exc,
			(	Exc =  error(existence_error(procedure, _), _)
			->	fail
			;	throw(Exc)
			)
		),
		(	Res = []
		->	true
		;	(	flag(debug)
			->	format(user_error, '.. eam/1 residual variables ~q left by premise ~q~n', [Res, Prem]),
				flush_output(user_error),
				fail
			)
		),
		(	(	Conc = false
			;	Conc = answer(false, void, void, _, _, _, _)
			)
		->	with_output_to(atom(PN3), wt('<http://www.w3.org/2000/10/swap/log#implies>'(Prem, false))),
			(	flag('ignore-inference-fuse')
			->	format(user_error, '** ERROR ** eam ** ~w~n', [inference_fuse(PN3)]),
				fail
			;	throw(inference_fuse(PN3))
			)
		;	true
		),
		\+atom(Conc),
		(	flag('rule-histogram'),
			copy_term_nat(Rule, RuleL)
		->	lookup(RTP, tp, RuleL),
			catch(cnt(RTP), _, nb_setval(RTP, 0))
		;	true
		),
		cnt(tp),
		(	(	flag(step, StepLim)	% DEPRECATED
			;	flag('limited-step', StepLim)
			),
			nb_getval(tp, Step),
			Step > StepLim
		->	(	flag(strings)
			->	true
			;	w3
			),
			(	flag(step, _)	% DEPRECATED
			->	throw(maximimum_step_count(Step))
			;	throw(halt)
			)
		;	true
		),
		djitin(Conc, Concdt),
		djitir(Concdt, Concdv),
		(	\+ground(Prem)
		->	clist(Lv, Concdv),
			partconc(Prem, Lv, Lw),
			clist(Lw, Concd)
		;	Concd = Concdv
		),
		(	flag(tactic, 'existing-path')
		->	makevars(Concd, Concdr, beta)
		;	Concdr = Concd
		),
		(	flag(think),	% DEPRECATED
			\+flag(nope),
			term_index(Prem, Pnd),
			term_index(Concdr, Cnd),
			prfstep(Concdr, Cnd, _, _, _, _, _, _),
			\+prfstep(_, _, Prem, Pnd, _, Rule, _, _)
		->	true
		;	\+call(Concdr)
		),
		(	flag('rule-histogram')
		->	lookup(RTC, tc, RuleL),
			catch(cnt(RTC), _, nb_setval(RTC, 0))
		;	true
		),
		(	Concd = cn(Cl)
		->	length(Cl, Ci),
			cnt(tc, Ci)
		;	cnt(tc)
		),
		(	Concd \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(_, _)
		->	nb_getval(wn, W),
			labelvars(Prem-Concd, W, N),
			nb_setval(wn, N)
		;	true
		),
		(	flag(debug)
		->	format(user_error, '... eam/1 assert step ~q~n', [Concd]),
			flush_output(user_error)
		;	true
		),
		clist(La, Concd),
		couple(La, La, Lc),
		findall([D, F],
			(	member([D, D], Lc),
				unify(D, F),
				(	flag(think),	% DEPRECATED
					\+flag(nope)
				->	true
				;	catch(\+call(F), _, true)
				)
			),
			Ld
		),
		couple(Ls, Le, Ld),
		clist(Ls, Concs),
		clist(Le, Conce),
		astep(Src, Prem, Concd, Conce, Rule),
		(	(	Concs = answer(_, _, _, _, _, _, _)
			;	Concs = cn([answer(_, _, _, _, _, _, _)|_])
			)
		->	cnt(answer_count)
		;	true
		),
		nb_getval(answer_count, AnswerCount),
		(	flag('limited-answer', AnswerLimit),
			AnswerCount >= AnswerLimit
		->	(	flag(strings)
			->	true
			;	w3
		)
		;	retract(brake),
			fail
		)
	;	(	brake
		;	flag(tactic, 'linear-select')
		),
		(	S is Span+1,
			(	\+span(S)
			->	assertz(span(S))
			;	true
			),
			nb_getval(limit, Limit),
			Span < Limit,
			eam(S)
		;	(	flag(strings)
			->	true
			;	w3
			)
		;	true
		),
		!
	;	assertz(brake),
		eam(Span)
	).


astep(A, B, Cd, Cn, Rule) :-	% astep(Source, Premise, Conclusion, Conclusion_unique, Rule)
	(	Cn = cn([Dn|En])
	->	functor(Dn, P, N),
		(	\+pred(P),
			P \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>',
			P \= '<http://www.w3.org/2000/10/swap/log#implies>',
			P \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>',
			N = 2
		->	assertz(pred(P))
		;	true
		),
		(	catch(call(Dn), _, fail)
		->	true
		;	djitis(Dn),
			(	flag('pass-only-new'),
				Dn \= answer(_, _, _, _, _, _, _)
			->	indent,
				relabel(Dn, Dr),
				wt(Dr),
				ws(Dr),
				write('.'),
				nl,
				cnt(output_statements)
			;	true
			)
		),
		(	flag(nope)
		->	true
		;	term_index(Dn, Cnd),
			term_index(B, Pnd),
			(	B = '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(P1, Q1),
				Rule = '<http://www.w3.org/2000/10/swap/log#implies>'(Q6, R6),
				prfstep('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(P1, Q1), _, Q3, Q4, _,
					'<http://www.w3.org/2000/10/swap/log#implies>'(P6, Q6), forward, A)
			->	(	\+prfstep(Dn, Cnd, Q3, Q4, Cd, '<http://www.w3.org/2000/10/swap/log#implies>'(P6, R6), forward, A)
				->	assertz(prfstep(Dn, Cnd, Q3, Q4, Cd, '<http://www.w3.org/2000/10/swap/log#implies>'(P6, R6), forward, A))
				;	true
				)
			;	(	\+prfstep(Dn, Cnd, B, Pnd, Cd, Rule, forward, A)
				->	assertz(prfstep(Dn, Cnd, B, Pnd, Cd, Rule, forward, A))
				;	true
				)
			)
		),
		(	En = [Fn]
		->	true
		;	Fn = cn(En)
		),
		astep(A, B, Cd, Fn, Rule)
	;	(	Cn = true
		->	true
		;	functor(Cn, P, N),
			(	\+pred(P),
				P \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>',
				P \= '<http://www.w3.org/2000/10/swap/log#implies>',
				P \= '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>',
				N = 2
			->	assertz(pred(P))
			;	true
			),
			(	catch(call(Cn), _, fail)
			->	true
			;	djitis(Cn),
				(	flag('pass-only-new'),
					Cn \= answer(_, _, _, _, _, _, _)
				->	indent,
					relabel(Cn, Cr),
					wt(Cr),
					ws(Cr),
					write('.'),
					nl,
					cnt(output_statements)
				;	true
				)
			),
			(	flag(nope)
			->	true
			;	term_index(Cn, Cnd),
				term_index(B, Pnd),
				(	B = '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(P1, Q1),
					Rule = '<http://www.w3.org/2000/10/swap/log#implies>'(Q6, R6),
					prfstep('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(P1, Q1), _, Q3, Q4, _,
						'<http://www.w3.org/2000/10/swap/log#implies>'(P6, Q6), forward, A)
				->	(	\+prfstep(Cn, Cnd, Q3, Q4, Cd, '<http://www.w3.org/2000/10/swap/log#implies>'(P6, R6), forward, A)
					->	assertz(prfstep(Cn, Cnd, Q3, Q4, Cd, '<http://www.w3.org/2000/10/swap/log#implies>'(P6, R6), forward, A))
					;	true
					)
				;	(	\+prfstep(Cn, Cnd, B, Pnd, Cd, Rule, forward, A)
					->	assertz(prfstep(Cn, Cnd, B, Pnd, Cd, Rule, forward, A))
					;	true
					)
				)
			)
		)
	).


istep(Src, Prem, Conc, Rule) :-		% istep(Source, Premise, Conclusion, Rule)
	copy_term_nat(Prem, Prec),
	labelvars(Prec, 0, _),
	term_index(Conc, Cnd),
	term_index(Prec, Pnd),
	(	\+prfstep(Conc, Cnd, Prec, Pnd, Conc, Rule, backward, Src)
	->	assertz(prfstep(Conc, Cnd, Prec, Pnd, Conc, Rule, backward, Src))
	;	true
	).


pstep(Rule) :-
	copy_term_nat(Rule, RuleL),
	lookup(RTC, tc, RuleL),
	catch(cnt(RTC), _, nb_setval(RTC, 0)),
	lookup(RTP, tp, RuleL),
	catch(cnt(RTP), _, nb_setval(RTP, 0)).


% DEPRECATED
hstep(A, B) :-
	(	nonvar(A),
		A = exopred(P, S, O)
	->	pred(P),
		U =.. [P, S, O],
		qstep(U, B)
	;	qstep(A, B)
	).


% DEPRECATED
qstep(A, B) :-
	prfstep(A, _, B, _, _, _, _, _).
% DEPRECATED
qstep(A, true) :-
	(	nonvar(A)
	->	(	A =.. [P, [S1, S2|S3], O]
		->	B =.. [P, S1, S2, S3, O]
		;	(	A =.. [P, S, literal(O1, O2)]
			->	B =.. [P, S, O1, O2]
			;	B = A
			)
		)
	;	pred(P),
		A =.. [P, _, _],
		B = A
	),
	catch(clause(B, true), _, fail),
	\+prfstep(A, _, _, _, _, _, _, _).


% DJITI (Deep Just In Time Indexing)

djiti(answer(cn(A)), cn(B)) :-
	!,
	djiti(A, B).
djiti(answer(A), answer(P, S, O, I, J, K, L)) :-
	(	nonvar(A)
	;	atom(P),
		S \= void
	),
	A =.. [P, S, O],
	!,
	(	var(I),
		compound(S)
	->	term_index(S, I),
		term_arg_1(S, K)
	;	true
	),
	(	var(J),
		compound(O)
	->	term_index(O, J),
		term_arg_1(O, L)
	;	true
	).
djiti(answer(exopred(P, S, O)), answer(P, S, O, I, J, K, L)) :-
	(	var(S)
	;	S \= void
	),
	!,
	(	var(I),
		compound(S)
	->	term_index(S, I),
		term_arg_1(S, K)
	;	true
	),
	(	var(J),
		compound(O)
	->	term_index(O, J),
		term_arg_1(O, L)
	;	true
	).
djiti(answer(A), answer(A, void, void, _, _, _, _)) :-
	!.
djiti([A|B], [C|D]) :-
	!,
	djiti(answer(A), C),
	djiti(B, D).
djiti(A, A).


djitin(answer(cn(A), void, void, _, _, _, _), cn(B)) :-
	!,
	djitin(A, B).
djitin([A|B], [answer(A, void, void, _, _, _, _)|C]) :-
	!,
	djitin(B, C).
djitin(A, A).


djitir(cn([A|B]), cn([C|D])) :-
	!,
	djitir(A, C),
	djitir(cn(B), cn(D)).
djitir(answer(P, S, O, _, _, _, _), answer(P, S, O, I, J, K, L)) :-
	!,
	term_index(S, I),
	term_index(O, J),
	term_arg_1(S, K),
	term_arg_1(O, L).
djitir(A, A).


djitis(answer(P, S, O, I, J, K, L)) :-
	atomic(P),
	!,
	(	current_predicate(P/7)
	->	true
	;	dynamic(P/7),
		assertz(':-'(answer(P, B2, B3, B4, B5, B6, B7),
				(	C =.. [P, B2, B3, B4, B5, B6, B7, answer],
					call(C)
				)
			)
		)
	),
	(	\+pred(P)
	->	assertz(pred(P))
	;	true
	),
	(	\+preda(P)
	->	assertz(preda(P))
	;	true
	),
	B =.. [P, S, O, I, J, K, L, answer],
	assertz(B).
djitis(exopred(P, S, O)) :-
	ground(exopred(P, S, O)),
	(	compound(S)
	;	compound(O)
	),
	!,
	term_index(S, Si),
	term_index(O, Oi),
	term_arg_1(S, Sp),
	term_arg_1(O, Op),
	(	current_predicate(exopred/7)
	->	true
	;	dynamic(exopred/7),
		assertz(':-'(exopred(P, U, V),
				(	(	compound(U)
					->	term_index(U, Ui),
						term_arg_1(U, Up)
					;	true
					),
					(	compound(V)
					->	term_index(V, Vi),
						term_arg_1(V, Vp)
					;	true
					),
					exopred(P, U, V, Ui, Vi, Up, Vp)
				)
			)
		)
	),
	assertz(exopred(P, S, O, Si, Oi, Sp, Op)).
djitis(A) :-
	ground(A),
	A =.. [P, S, O],
	A \= ':-'(_, _),
	(	compound(S)
	;	compound(O)
	),
	!,
	term_index(S, Si),
	term_index(O, Oi),
	term_arg_1(S, Sp),
	term_arg_1(O, Op),
	(	current_predicate(P/6)
	->	true
	;	dynamic(P/6),
		X =.. [P, U, V],
		assertz(':-'(X,
				(	(	compound(U)
					->	term_index(U, Ui),
						term_arg_1(U, Up)
					;	true
					),
					(	compound(V)
					->	term_index(V, Vi),
						term_arg_1(V, Vp)
					;	true
					),
					Y =.. [P, U, V, Ui, Vi, Up, Vp],
					call(Y)
				)
			)
		)
	),
	B =.. [P, S, O, Si, Oi, Sp, Op],
	assertz(B).
djitis('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'(A, B)) :-
	!,
	(	current_predicate(type_index/3)
	->	true
	;	dynamic(type_index/3),
		assertz(':-'('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'(U, V),
				(	term_index(U-V, W),
					type_index(U, V, W)
				)
			)
		)
	),
	term_index(A-B, C),
	assertz(type_index(A, B, C)).
djitis(A) :-
	assertz(A).


% Built-ins

'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#biconditional>'(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, B)|C], D) :-
	within_scope(_),
	(	nb_getval(bnet, done)
	->	true
	;	bnet,
		nb_setval(bnet, done)
	),
	bvar(A),
	bval(B),
	bcon(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, B)], C, D).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#binaryEntropy>'(A, B) :-
	getnumber(A, C),
	(	C =:= 0.0
	->	B is 0.0
	;	(	C =:= 1.0
		->	B is 0.0
		;	B is -(C*log(C)+(1-C)*log(1-C))/log(2)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#calculate>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))|B], C) :-
	findall(U,
		(	member(V, B),
			getnumber(V, U)
		),
		W
	),
	read_term_from_atom(A, D, [variables(W)]),
	catch(C is D, _, fail).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#call>'(Sc, A) :-
	within_scope(Sc),
	nonvar(A),
	call(A),
	(	flag(nope)
	->	true
	;	copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#call>'(Sc, A)), B),
		istep('<>', A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#call>'(Sc, A), B)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#cartesianProduct>'(A, B) :-
	findall(C,
		(	cartesian(A, C)
		),
		B
	).


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#closure>'(Sc, A) :-
	within_scope(Sc),
	hstep(A, _).


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#distinct>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	distinct(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#fail>'(A, B) :-
	within_scope(A),
	\+call(B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#finalize>'(A, B) :-
	call_cleanup(A, B),
	(	flag(nope)
	->	true
	;	clist(C, A),
		clist(D, B),
		append(C, D, E),
		clist(E, F),
		copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(F, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#finalize>'(A, B)), G),
		istep('<>', F, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#finalize>'(A, B), G)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#findall>'(Sc, [A, B, C|D]) :-
	within_scope(Sc),
	nonvar(B),
	\+is_list(B),
	(	D = [F]
	->	findall(A,	% DEPRECATED
			B,
			E,
			F
		)
	;	findall(A,
			B,
			E
		)
	),
	(	flag(warn)
	->	copy_term_nat([A, B, E|D], [Ac, Bc, Ec|Dc]),
		labelvars([Ac, Bc, Ec|Dc], 0, _),
		(	fact('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#findall>'(Sc, [Ac, Bc, G|H]))
		->	(	E \= G
			->	format(user_error, '** WARNING ** conflicting_findall_answers ~w VERSUS ~w~n', [[A, B, G|H], [A, B, E|D]]),
				flush_output(user_error)
			;	true
			)
		;	assertz(fact('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#findall>'(Sc, [Ac, Bc, Ec|Dc])))
		)
	;	true
	),
	E = C.


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#format>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))|B], literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, D),
			preformat(B, E),
			format_to_chars(D, E, F),
			atom_codes(C, F)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#graphCopy>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#labelvars>'(A, C),
			clist(L, C),
			sort(L, M),
			clist(M, K),
			unify(K, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#graphDifference>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	difference(A, M),
			unify(M, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#graphIntersection>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	intersect(A, M),
			unify(M, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#graphList>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	clistflat(B, A)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#hmac-sha>'(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	flag('hmac-key', Key),
	hmac_sha(Key, A, C, [algorithm(sha1)]),
	atom_codes(D, C),
	base64xml(D, B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#ignore>'(Sc, A) :-
	within_scope(Sc),
	nonvar(A),
	(	call(A)
	->	(	flag(nope)
		->	true
		;	copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#ignore>'(Sc, A)), R),
			istep('<>', A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#ignore>'(Sc, A), R)
		)
	;	true
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#label>'(A, literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	nonvar(A)
		),
		(	atom(A),
			(	sub_atom(A, _, 19, _, '/.well-known/genid/')
			->	(	sub_atom(A, I, 1, _, '#')
				->	J is I+1,
					sub_atom(A, J, _, 1, B)
				;	B = ''
				)
			;	atom_concat(some, C, A),
				atomic_list_concat(['sk_', C], B)
			)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#labelvars>'(A, B) :-
	(	got_labelvars(A, B)
	->	true
	;	copy_term_nat(A, B),
		nb_getval(wn, W),
		labelvars(B, W, N),
		nb_setval(wn, N),
		assertz(got_labelvars(A, B))
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#length>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	(	getlist(A, C)
			->	true
			;	clistflat(C, A)
			),
			length(C, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#match>'(_, B) :-
	\+ \+call(B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#max>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	bmax(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#min>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	bmin(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#notLabel>'(A, B) :-
	\+'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#label>'(A, B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#numeral>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getnumber(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#optional>'(Sc, A) :-
	within_scope(Sc),
	nonvar(A),
	(	\+call(A)
	->	true
	;	call(A),
		(	flag(nope)
		->	true
		;	copy_term_nat('<http://www.w3.org/2000/10/swap/log#implies>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#optional>'(Sc, A)), R),
			istep('<>', A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#optional>'(Sc, A), R)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#prefix>'(Sc, literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	within_scope(Sc),
	with_output_to_codes(wh, C),
	atom_codes(A, C),
	retractall(wpfx(_)).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#propertyChainExtension>'([A], [B, C]) :-
	!,
	D =.. [A, B, C],
	catch(call(D), _, fail).
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#propertyChainExtension>'([A|B], [C, D]) :-
	E =.. [A, C, F],
	catch(call(E), _, fail),
	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#propertyChainExtension>'(B, [F, D]).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#random>'([A|B], C) :-
	term_index([A|B], I),
	(	B \= [],
		got_random([A|B], I, C)
	->	true
	;	catch(nb_getval(random, D), _, D = 1298074214633706835075030044377087),
		E is mod(19134702400093278081449423917*D+359334085968622831041960188598043661065388726959079837, 43143988327398957279342419750374600193),
		nb_setval(random, E),
		C is mod(E, A),
		(	B \= []
		->	assertz(got_random([A|B], I, C))
		;	true
		)
	).


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#reason>'(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), B) :-
	when(
		(	ground(A)
		),
		(	sub_atom(A, 0, 4, _, 'eye '),
			sub_atom(A, 4, _, 0, C),
			(	current_prolog_flag(windows, true)
			->	A1 = ['cmd.exe', '/C']
			;	A1 = []
			),
			(	current_prolog_flag(argv, Argv),
				append(Argu, ['--'|_], Argv)
			->	append(Argu, ['--'], A2)
			;	A2 = ['eye']
			),
			append([A1, A2, [C]], A4),
			findall([G, ' '],
				(	member(G, A4)
				),
				H
			),
			flatten(H, I),
			atomic_list_concat(I, J),
			exec(J, B)
		)
	).


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#reverse>'(A, B) :-
	reverse(A, B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#roc>'(St, [Sen, Asp]) :-
	getnumber(St, K),
	(	getnumber(Sen, S)
	->	Asp is 1-(1-exp(-K*(S-1)))*(1+exp(K))/(1+exp(-K*(S-1)))/(1-exp(K))
	;	getnumber(Asp, A),
		Sen is (1-exp(-K*A))*(1+exp(-K))/(1+exp(-K*A))/(1-exp(-K))
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#sha>'(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	sha_hash(A, C, [algorithm(sha1)]),
	atom_codes(D, C),
	base64xml(D, B).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#sigmoid>'(A, B) :-
	getnumber(A, C),
	B is 1/(1+exp(-C)).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#skolem>'(X, Y) :-
	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tuple>'(Y, X),
	(	\+keep_skolem(Y)
	->	assertz(keep_skolem(Y))
	;	true
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#sort>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	quicksort(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#stringEscape>'(literal(X, Y), literal(Z, Y)) :-
	when(
		(	ground(X)
		),
		(	atom_codes(X, U),
			escape_string(U, V),
			atom_codes(Z, V)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#stringSplit>'([literal(X, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	Z) :-
	when(
		(	ground([X, Y])
		),
		(	atom_codes(X, U),
			atom_codes(Y, V),
			split_string(U, V, "", W),
			findall(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')),
				(	member(B, W),
					atom_codes(A, B)
				),
				Z
			)
		)
	).


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#sublist>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	sub_list(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#subsequence>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	sub_list(A, B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#trace>'(X, Y) :-
	tell(user_error),
	write('TRACE '),
	(	var(X)
	->	ignore(get_time(X)),
		wg(Y)
	;	writeq(Y)
	),
	nl,
	told.


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tripleList>'(A, [B, C, D]) :-
	A =.. [C, B, D].


% DEPRECATED
'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#true>'(_, A) :-
	when(
		(	nonvar(A)
		),
		(	A =:= 1.0
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tuple>'(X, Y) :-
	when(
		(	nonvar(X)
		;	ground(Y)
		),
		(	(	is_list(Y),
				length(Y, I),
				I < 8
			->	Z =.. [tuple, X|Y]
			;	Z =.. [tuple, X, Y]
			),
			(	call(Z)
			->	true
			;	var(X),
				nb_getval(tuple, M),
				N is M+1,
				nb_setval(tuple, N),
				atom_number(A, N),
				nb_getval(var_ns, Vns),
				atomic_list_concat(['<', Vns, 't_', A, '>'], X),
				assertz(Z)
			)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#unique>'(A, B) :-
	(	got_unique(A, B)
	->	fail
	;	assertz(got_unique(A, B))
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#whenGround>'(A, B) :-
	when(
		(	ground(A)
		),
		(	findvars(A, C, delta),
			C \= []
		->	true
		;	catch(call(B), _, A = B)
		)
	).


'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#wwwFormEncode>'(X, literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	(	ground(X)
			->	(	number(X)
				->	atom_number(T, X)
				;	X = literal(T, _)
				),
				www_form_encode(T, Z),
				atom_codes(Z, U),
				subst([[[0'%, 0'2, 0'0], [0'+]]], U, V),
				atom_codes(Y, V)
			;	www_form_encode(X, Y)
			)
		)
	).


% DEPRECATED
'<http://www.w3.org/2005/xpath-functions#resolve-uri>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	resolve_uri(A, B, C)
		)
	).


% DEPRECATED
'<http://www.w3.org/2005/xpath-functions#substring>'([literal(A, _), B|C], literal(D, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B, C])
		),
		(	atom_codes(A, U),
			(	C = []
			->	length(U, E),
				F is E-B
			;	C = [F]
			),
			sub_atom(A, B, F, _, D)
		)
	).


% DEPRECATED
'<http://www.w3.org/2005/xpath-functions#substring-after>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, _, _, W, B),
			sub_atom(A, _, W, 0, C)
		)
	).


% DEPRECATED
'<http://www.w3.org/2005/xpath-functions#substring-before>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, W, _, _, B),
			sub_atom(A, 0, W, _, C)
		)
	).


'<http://www.w3.org/2000/10/swap/crypto#sha>'(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	sha_hash(A, C, [algorithm(sha1)]),
	hash_to_ascii(C, D, []),
	atom_codes(B, D).


'<http://www.w3.org/2000/10/swap/list#append>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getlist(A, C),
			append(C, B)
		)
	).


'<http://www.w3.org/2000/10/swap/list#first>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getlist(A, C),
			C = [B|D],
			nonvar(D)
		)
	).


'<http://www.w3.org/2000/10/swap/list#in>'(A, B) :-
	when(
		(	nonvar(B)
		),
		(	getlist(B, C),
			member(A, C)
		)
	).


'<http://www.w3.org/2000/10/swap/list#last>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getlist(A, C),
			last(C, B)
		)
	).


'<http://www.w3.org/2000/10/swap/list#member>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getlist(A, C),
			member(B, C)
		)
	).


'<http://www.w3.org/2000/10/swap/list#rest>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	getlist(A, C),
			C = [_|B]
		)
	).


'<http://www.w3.org/2000/10/swap/log#conclusion>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	reset_gensym,
			tmp_file(Tmp1),
			open(Tmp1, write, Ws1, [encoding(utf8)]),
			tell(Ws1),
			(	flag('no-qnames')
			->	true
			;	forall(
					(	pfx(C, D)
					),
					(	(	\+flag(traditional)
						->	format('PREFIX ~w ~w~n', [C, D])
						;	format('@prefix ~w ~w.~n', [C, D])
						)
					)
				),
				nl
			),
			labelvars(A, 0, _),
			wt(A),
			write('.'),
			nl,
			told,
			tmp_file(Tmp2),
			!,
			(	current_prolog_flag(windows, true)
			->	A1 = ['cmd.exe', '/C']
			;	A1 = []
			),
			(	current_prolog_flag(argv, Argv),
				append(Argu, ['--'|_], Argv)
			->	append(Argu, ['--'], A2)
			;	A2 = ['eye']
			),
			append([A1, A2, ['--nope', Tmp1, '--pass-all', '>', Tmp2]], A4),
			findall([G, ' '],
				(	member(G, A4)
				),
				H
			),
			flatten(H, I),
			atomic_list_concat(I, J),
			(	catch(exec(J, _), _, fail)
			->	n3_n3p(Tmp2, semantics),
				absolute_uri(Tmp2, Tmp),
				atomic_list_concat(['<', Tmp, '>'], Res),
				semantics(Res, B),
				labelvars(B, 0, _),
				delete_file(Tmp1),
				delete_file(Tmp2)
			;	delete_file(Tmp1),
				delete_file(Tmp2),
				fail
			)
		)
	).


'<http://www.w3.org/2000/10/swap/log#conjunction>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	conjoin(A, M),
			unify(M, B)
		)
	).


'<http://www.w3.org/2000/10/swap/log#dtlit>'([A, B], C) :-
	when(
		(	ground(A)
		;	nonvar(C)
		),
		(	ground(A),
			(	var(B)
			->	(	member(B, ['<http://www.w3.org/2001/XMLSchema#integer>', '<http://www.w3.org/2001/XMLSchema#double>',
					'<http://www.w3.org/2001/XMLSchema#date>', '<http://www.w3.org/2001/XMLSchema#time>', '<http://www.w3.org/2001/XMLSchema#dateTime>',
					'<http://www.w3.org/2001/XMLSchema#yearMonthDuration>', '<http://www.w3.org/2001/XMLSchema#dayTimeDuration>', '<http://www.w3.org/2001/XMLSchema#duration>']),
					dtlit([A, B], C),
					getnumber(C, D),
					dtlit([_, B], D)
				->	true
				;	(	dtlit([A, '<http://www.w3.org/2001/XMLSchema#boolean>'], C),
						getbool(C, _),
						B = '<http://www.w3.org/2001/XMLSchema#boolean>'
					->	true
					;	B = '<http://www.w3.org/2001/XMLSchema#string>',
						C = A
					)
				)
			;	A = literal(E, _),
				(	B = prolog:atom
				->	C = E
				;	C = literal(E, type(B))
				),
				!
			)
		;	nonvar(C),
			dtlit([A, B], C)
		)
	).


'<http://www.w3.org/2000/10/swap/log#equalTo>'(X, Y) :-
	unify(X, Y).


'<http://www.w3.org/2000/10/swap/log#implies>'(X, Y) :-
	implies(X, Z, _),
	(	commonvars(X, Z, [])
	->	labelvars(Z, 0, _, avar)
	;	true
	),
	Y = Z,
	Y \= answer(_, _, _, _, _, _, _),
	Y \= cn([answer(_, _, _, _, _, _, _)|_]).


'<http://www.w3.org/2000/10/swap/log#includes>'(X, Y) :-
	when(
		(	nonvar(X),
			nonvar(Y)
		),
		(	catch(cnt(graph), _, nb_setval(graph, 0)),
			nb_getval(graph, N),
			copy_term_nat(X, U),
			makevars(Y, V, beta),
			agraph(N, U),
			qgraph(N, V)
		)
	).


'<http://www.w3.org/2000/10/swap/log#rawType>'(A, B) :-
	nonvar(A),
	raw_type(A, C),
	C = B.


'<http://www.w3.org/2000/10/swap/log#notEqualTo>'(X, Y) :-
	\+'<http://www.w3.org/2000/10/swap/log#equalTo>'(X, Y).


'<http://www.w3.org/2000/10/swap/log#notIncludes>'(X, Y) :-
	when(
		(	nonvar(X),
			nonvar(Y)
		),
		(	catch(cnt(graph), _, nb_setval(graph, 0)),
			nb_getval(graph, N),
			copy_term_nat(X, U),
			labelvars(U, 0, _),
			makevars(Y, V, beta),
			agraph(N, U),
			\+qgraph(N, V)
		)
	).


'<http://www.w3.org/2000/10/swap/log#semantics>'(X, Y) :-
	when(
		(	nonvar(X)
		),
		(	(	semantics(X, Q)
			->	Y = Q
			;	sub_atom(X, 0, 1, _, '<'),
				sub_atom(X, _, 1, 0, '>'),
				sub_atom(X, 1, _, 1, Z),
				catch(n3_n3p(Z, semantics), Exc,
					(	format(user_error, '** ERROR ** ~w **~n', [Exc]),
						flush_output(user_error),
						fail
					)
				),
				semantics(X, Y)
			)
		)
	).


'<http://www.w3.org/2000/10/swap/log#uri>'(X, Y) :-
	when(
		(	nonvar(X)
		;	nonvar(Y)
		),
		(	atomic(X),
			(	atom_concat(some, V, X)
			->	nb_getval(var_ns, Vns),
				atomic_list_concat(['<', Vns, 'sk_', V, '>'], U)
			;	U = X
			),
			sub_atom(U, 1, _, 1, Z),
			atomic_list_concat(['<', Z, '>'], U),
			Y = literal(Z, type('<http://www.w3.org/2001/XMLSchema#string>')),
			!
		;	nonvar(Y),
			Y = literal(Z, type('<http://www.w3.org/2001/XMLSchema#string>')),
			atomic_list_concat(['<', Z, '>'], X)
		)
	).


'<http://www.w3.org/2000/10/swap/math#absoluteValue>'(X, Y) :-
	when(
		(	ground(X)
		),
		(	getnumber(X, U),
			Y is abs(U)
		)
	).


'<http://www.w3.org/2000/10/swap/math#atan2>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			Z is atan(U/V)
		)
	).


'<http://www.w3.org/2000/10/swap/math#cos>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is cos(U),
			!
		;	getnumber(Y, W),
			X is acos(W)
		)
	).


'<http://www.w3.org/2000/10/swap/math#cosh>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is cosh(U),
			!
		;	getnumber(Y, W),
			X is acosh(W)
		)
	).


'<http://www.w3.org/2000/10/swap/math#degrees>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is U*180/pi,
			!
		;	getnumber(Y, W),
			X is W*pi/180
		)
	).


'<http://www.w3.org/2000/10/swap/math#difference>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			Z is U-V
		)
	).


'<http://www.w3.org/2000/10/swap/math#equalTo>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U =:= V
		)
	).


'<http://www.w3.org/2000/10/swap/math#exponentiation>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		;	ground([X, Z])
		),
		(	getnumber(X, U),
			(	getnumber(Y, V),
				Z is U**V,
				!
			;	getnumber(Z, W),
				W =\= 0,
				U =\= 0,
				Y is log(W)/log(U)
			)
		)
	).


'<http://www.w3.org/2000/10/swap/math#greaterThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U > V
		)
	).


'<http://www.w3.org/2000/10/swap/math#integerQuotient>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			(	V =\= 0
			->	Z is round(floor(U/V))
			;	throw(zero_division('<http://www.w3.org/2000/10/swap/math#integerQuotient>'([X, Y], Z)))
			)
		)
	).


'<http://www.w3.org/2000/10/swap/math#lessThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U < V
		)
	).


'<http://www.w3.org/2000/10/swap/math#memberCount>'(X, Y) :-
	when(
		(	nonvar(X)
		),
		(	(	getlist(X, Z)
			->	true
			;	clistflat(Z, X)
			),
			length(Z, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/math#negation>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is -U,
			!
		;	getnumber(Y, W),
			X is -W
		)
	).


'<http://www.w3.org/2000/10/swap/math#notEqualTo>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U =\= V
		)
	).


'<http://www.w3.org/2000/10/swap/math#notGreaterThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U =< V
		)
	).


'<http://www.w3.org/2000/10/swap/math#notLessThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			U >= V
		)
	).


'<http://www.w3.org/2000/10/swap/math#product>'(X, Y) :-
	when(
		(	ground(X)
		),
		(	product(X, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/math#quotient>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			(	V =\= 0
			->	Z is U/V
			;	throw(zero_division('<http://www.w3.org/2000/10/swap/math#quotient>'([X, Y], Z)))
			)
		)
	).


'<http://www.w3.org/2000/10/swap/math#remainder>'([X, Y], Z) :-
	when(
		(	ground([X, Y])
		),
		(	getnumber(X, U),
			getnumber(Y, V),
			(	V =\= 0
			->	Z is U-V*round(floor(U/V))
			;	throw(zero_division('<http://www.w3.org/2000/10/swap/math#remainder>'([X, Y], Z)))
			)
		)
	).


'<http://www.w3.org/2000/10/swap/math#rounded>'(X, Y) :-
	when(
		(	ground(X)
		),
		(	getnumber(X, U),
			Y is round(round(U))
		)
	).


'<http://www.w3.org/2000/10/swap/math#sin>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is sin(U),
			!
		;	getnumber(Y, W),
			X is asin(W)
		)
	).


'<http://www.w3.org/2000/10/swap/math#sinh>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is sinh(U),
			!
		;	getnumber(Y, W),
			X is asinh(W)
		)
	).


'<http://www.w3.org/2000/10/swap/math#sum>'(X, Y) :-
	when(
		(	ground(X)
		),
		(	sum(X, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/math#tan>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is tan(U),
			!
		;	getnumber(Y, W),
			X is atan(W)
		)
	).


'<http://www.w3.org/2000/10/swap/math#tanh>'(X, Y) :-
	when(
		(	ground(X)
		;	ground(Y)
		),
		(	getnumber(X, U),
			Y is tanh(U),
			!
		;	getnumber(Y, W),
			X is atanh(W)
		)
	).


'<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>'(X, Y) :-
	when(
		(	nonvar(X)
		),
		(	X = [Y|Z],
			nonvar(Z)
		)
	),
	!.


'<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>'(X, Y) :-
	when(
		(	nonvar(X)
		),
		(	X = [_|Y]
		)
	),
	!.


'<http://www.w3.org/2000/10/swap/string#concatenation>'(X, literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	nonvar(X)
		),
		(	findall(S,
				(	member(A, X),
					getcodes(A, S)
				),
				Z
			),
			flatten(Z, C),
			atom_codes(Y, C)
		)
	).


'<http://www.w3.org/2000/10/swap/string#contains>'(literal(X, Z), literal(Y, Z)) :-
	when(
		(	ground([X, Y])
		),
		(	sub_atom(X, _, _, _, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/string#containsIgnoringCase>'(literal(X, Z), literal(Y, Z)) :-
	when(
		(	ground([X, Y])
		),
		(	downcase_atom(X, U),
			downcase_atom(Y, V),
			sub_atom(U, _, _, _, V)
		)
	).


'<http://www.w3.org/2000/10/swap/string#endsWith>'(literal(X, _), literal(Y, _)) :-
	when(
		(	ground([X, Y])
		),
		(	sub_atom(X, _, _, 0, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/string#equalIgnoringCase>'(literal(X, _), literal(Y, _)) :-
	when(
		(	ground([X, Y])
		),
		(	downcase_atom(X, U),
			downcase_atom(Y, U)
		)
	).


'<http://www.w3.org/2000/10/swap/string#greaterThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getstring(X, U),
			getstring(Y, V),
			U @> V
		)
	).


'<http://www.w3.org/2000/10/swap/string#lessThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getstring(X, U),
			getstring(Y, V),
			U @< V
		)
	).


'<http://www.w3.org/2000/10/swap/string#matches>'(literal(X, _), literal(Y, _)) :-
	when(
		(	ground([X, Y])
		),
		(	atom_codes(X, U),
			atom_codes(Y, V),
			regex(V, U, _)
		)
	).


'<http://www.w3.org/2000/10/swap/string#notEqualIgnoringCase>'(X, Y) :-
	\+'<http://www.w3.org/2000/10/swap/string#equalIgnoringCase>'(X, Y).


'<http://www.w3.org/2000/10/swap/string#notGreaterThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getstring(X, U),
			getstring(Y, V),
			U @=< V
		)
	).


'<http://www.w3.org/2000/10/swap/string#notLessThan>'(X, Y) :-
	when(
		(	ground([X, Y])
		),
		(	getstring(X, U),
			getstring(Y, V),
			U @>= V
		)
	).


'<http://www.w3.org/2000/10/swap/string#notMatches>'(X, Y) :-
	\+'<http://www.w3.org/2000/10/swap/string#matches>'(X, Y).


'<http://www.w3.org/2000/10/swap/string#replace>'([literal(X, _), literal(Search, _), literal(Replace, _)], literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([X, Search, Replace])
		),
		(	atomic_list_concat(Split, Search, X),
			atomic_list_concat(Split, Replace, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/string#scrape>'([literal(X, _), literal(Y, _)], literal(Z, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([X, Y])
		),
		(	atom_codes(X, U),
			atom_codes(Y, V),
			regex(V, U, [W|_]),
			atom_codes(Z, W)
		)
	).


'<http://www.w3.org/2000/10/swap/string#search>'([literal(X, _), literal(Y, _)], Z) :-
	when(
		(	ground([X, Y])
		),
		(	atom_codes(X, U),
			atom_codes(Y, V),
			regex(V, U, L),
			findall(literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')),
				(	member(M, L),
					atom_codes(A, M)
				),
				Z
			)
		)
	).


'<http://www.w3.org/2000/10/swap/string#startsWith>'(literal(X, _), literal(Y, _)) :-
	when(
		(	ground([X, Y])
		),
		(	sub_atom(X, 0, _, _, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/time#day>'(literal(X, _), literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground(X)
		),
		(	sub_atom(X, 8, 2, _, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/time#month>'(literal(X, _), literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground(X)
		),
		(	sub_atom(X, 5, 2, _, Y)
		)
	).


'<http://www.w3.org/2000/10/swap/time#year>'(literal(X, _), literal(Y, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground(X)
		),
		(	sub_atom(X, 0, 4, _, Y)
		)
	).


% RIF built-ins according to RIF Datatypes and Built-Ins 1.0 -- http://www.w3.org/TR/rif-dtb/

% 4.1.1.1 pred:literal-not-identical

'<http://www.w3.org/2007/rif-builtin-predicate#literal-not-identical>'([literal(A, B), literal(C, B)], D) :-
	when(
		(	ground([A, B, C])
		),
		(	A \== C
		->	D = true
		;	D = false
		)
	).


% 4.4.4 pred:iri-string

'<http://www.w3.org/2007/rif-builtin-predicate#iri-string>'([A, literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	when(
		(	nonvar(A)
		;	nonvar(B)
		),
		(	atom(A),
			sub_atom(A, 1, _, 1, U),
			atomic_list_concat(['<', U, '>'], A),
			!,
			(	U = B
			->	C = true
			;	C = false
			)
		;	nonvar(B),
			(	atomic_list_concat(['<', B, '>'], A)
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.1 Numeric Functions

'<http://www.w3.org/2007/rif-builtin-function#numeric-add>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	sum([A, B], C)
		)
	).


'<http://www.w3.org/2007/rif-builtin-function#numeric-subtract>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			C is U-V
		)
	).


'<http://www.w3.org/2007/rif-builtin-function#numeric-multiply>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			C is U*V
		)
	).


'<http://www.w3.org/2007/rif-builtin-function#numeric-divide>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	V =\= 0
			->	C is U/V
			;	throw(zero_division('<http://www.w3.org/2007/rif-builtin-function#numeric-divide>'([A, B], C)))
			)
		)
	).


'<http://www.w3.org/2007/rif-builtin-function#numeric-integer-divide>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	V =\= 0
			->	C is integer(floor(U/V))
			;	throw(zero_division('<http://www.w3.org/2007/rif-builtin-function#numeric-integer-divide>'([A, B], C)))
			)
		)
	).


'<http://www.w3.org/2007/rif-builtin-function#numeric-mod>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	V =\= 0
			->	C is U-V*integer(floor(U/V))
			;	throw(zero_division('<http://www.w3.org/2007/rif-builtin-function#numeric-mod>'([A, B], C)))
			)
		)
	).


% 4.5.2.1 pred:numeric-equal

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-equal>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U =:= V
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.2.2 pred:numeric-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-less-than>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U < V
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.2.3 pred:numeric-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-greater-than>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U > V
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.2.4 pred:numeric-not-equal

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-not-equal>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U =\= V
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.2.5 pred:numeric-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-less-than-or-equal>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U =< V
			->	C = true
			;	C = false
			)
		)
	).


% 4.5.2.6 pred:numeric-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#numeric-greater-than-or-equal>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getnumber(A, U),
			getnumber(B, V),
			(	U >= V
			->	C = true
			;	C = false
			)
		)
	).


% 4.6.1.1 func:not

'<http://www.w3.org/2007/rif-builtin-function#not>'([A], B) :-
	when(
		(	ground(A)
		),
		(	getbool(A, U),
			(	ground(B)
			->	getbool(B, V)
			;	V = B
			),
			inv(U, V)
		)
	).


% 4.6.2.1 pred:boolean-equal

'<http://www.w3.org/2007/rif-builtin-predicate#boolean-equal>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getbool(A, U),
			getbool(B, U)
		->	C = true
		;	C = false
		)
	).


% 4.6.2.2 pred:boolean-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#boolean-less-than>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getbool(A, false),
			getbool(B, true)
		->	C = true
		;	C = false
		)
	).


% 4.6.2.3 pred:boolean-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#boolean-greater-than>'([A, B], C) :-
	when(
		(	ground([A, B])
		),
		(	getbool(A, true),
			getbool(B, false)
		->	C = true
		;	C = false
		)
	).


% 4.7.1.1 func:compare @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-function#compare>'([literal(A, B), literal(C, B)], D) :-
	!,
	(	A @< C
	->	D = -1
	;	(	A == C
		->	D = 0
		;	(	A @> C
			->	D = 1
			)
		)
	).


% 4.7.1.2 func:concat

'<http://www.w3.org/2007/rif-builtin-function#concat>'(A, literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground(A)
		),
		(	findall(F,
				(	member(literal(S, type('<http://www.w3.org/2001/XMLSchema#string>')), A),
					atom_codes(S, F)
				),
				C
			),
			flatten(C, D),
			atom_codes(B, D)
		)
	).


% 4.7.1.3 func:string-join

'<http://www.w3.org/2007/rif-builtin-function#string-join>'([A, literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(B, D),
			findall([D, E],
				(	member(literal(F, type('<http://www.w3.org/2001/XMLSchema#string>')), A),
					atom_codes(F, E)
				),
				G
			),
			(	G = [[_, H]|I]
			->	flatten([H|I], J),
				atom_codes(C, J)
			;	C = ''
			)
		)
	).


% 4.7.1.4 func:substring

'<http://www.w3.org/2007/rif-builtin-function#substring>'([literal(A, _), B, C], literal(D, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	!,
	when(
		(	ground([A, B, C])
		),
		(	getint(B, I),
			getint(C, J),
			(	I < 1
			->	G is 0,
				H is J+I-1
			;	G is I-1,
				H is J
			),
			(	H < 0
			->	D = ''
			;	sub_atom(A, G, H, _, D)
			)
		)
	).
'<http://www.w3.org/2007/rif-builtin-function#substring>'([literal(A, _), B], literal(D, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	getint(B, I),
			sub_atom(A, 0, E, 0, _),
			J is E-I+1,
			(	I < 1
			->	G is 0,
				H is J+I-1
			;	G is I-1,
				H is J
			),
			(	H < 0
			->	D = []
			;	sub_atom(A, G, H, _, D)
			)
		)
	).


% 4.7.1.5 func:string-length

'<http://www.w3.org/2007/rif-builtin-function#string-length>'([literal(A, _)], B) :-
	when(
		(	ground(A)
		),
		(	sub_atom(A, 0, B, 0, _)
		)
	).


% 4.7.1.6 func:upper-case

'<http://www.w3.org/2007/rif-builtin-function#upper-case>'([literal(A, B)], literal(C, B)) :-
	when(
		(	ground([A, B])
		),
		(	upcase_atom(A, C)
		)
	).


% 4.7.1.7 func:lower-case

'<http://www.w3.org/2007/rif-builtin-function#lower-case>'([literal(A, B)], literal(C, B)) :-
	when(
		(	ground([A, B])
		),
		(	downcase_atom(A, C)
		)
	).


% 4.7.1.8 func:encode-for-uri

'<http://www.w3.org/2007/rif-builtin-function#encode-for-uri>'([literal(A, B)], literal(C, B)) :-
	when(
		(	ground([A, B])
		),
		(	www_form_encode(A, C)
		)
	).


% 4.7.1.11 func:substring-before @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-function#substring-before>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, W, _, _, B),
			sub_atom(A, 0, W, _, C)
		)
	).


% 4.7.1.12 func:substring-after @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-function#substring-after>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, _, _, W, B),
			sub_atom(A, _, W, 0, C)
		)
	).


% 4.7.2.1 pred:contains @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-predicate#contains>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(B, 0, D, 0, _),
			sub_atom(A, _, D, _, B)
		->	C = true
		;	C = false
		)
	).


% 4.7.2.2 pred:starts-with @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-predicate#starts-with>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, 0, _, _, B)
		->	C = true
		;	C = false
		)
	).


% 4.7.2.3 pred:ends-with @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-predicate#ends-with>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	sub_atom(A, _, _, 0, B)
		->	C = true
		;	C = false
		)
	).


% 4.7.2.4 pred:matches @@partial implementation: no flags

'<http://www.w3.org/2007/rif-builtin-predicate#matches>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			regex(V, U, _)
		->	C = true
		;	C = false
		)
	).


% 4.8.1.1 func:year-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#year-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(C, _, _, _, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.2 func:month-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#month-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, C, _, _, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.3 func:day-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#day-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, _, C, _, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.4 func:hours-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#hours-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, _, _, C, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.5 func:minutes-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#minutes-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, _, _, _, C, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.6 func:seconds-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#seconds-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, _, _, _, _, C, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.7 func:year-from-date

'<http://www.w3.org/2007/rif-builtin-function#year-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			date(C, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.8 func:month-from-date

'<http://www.w3.org/2007/rif-builtin-function#month-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			date(_, C, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.9 func:day-from-date

'<http://www.w3.org/2007/rif-builtin-function#day-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			date(_, _, C, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.10 func:hours-from-time

'<http://www.w3.org/2007/rif-builtin-function#hours-from-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			time(C, _, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.11 func:minutes-from-time

'<http://www.w3.org/2007/rif-builtin-function#minutes-from-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			time(_, C, _, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.12 func:seconds-from-time

'<http://www.w3.org/2007/rif-builtin-function#seconds-from-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			time(_, _, C, _, U, []),
			(	nonvar(B)
			->	C =:= B
			;	C = B
			)
		)
	).


% 4.8.1.13 func:years-from-duration

'<http://www.w3.org/2007/rif-builtin-function#years-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], 0) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#years-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			yearmonthduration(C, U, []),
			D is C//12,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.14 func:months-from-duration

'<http://www.w3.org/2007/rif-builtin-function#months-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], 0) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#months-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			yearmonthduration(C, U, []),
			D is C-(C//12)*12,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.15 func:days-from-duration

'<http://www.w3.org/2007/rif-builtin-function#days-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], _) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#days-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			daytimeduration(C, U, []),
			D is integer(C)//86400,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.16 func:hours-from-duration

'<http://www.w3.org/2007/rif-builtin-function#hours-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], _) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#hours-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			daytimeduration(C, U, []),
			D is (integer(C)-(integer(C)//86400)*86400)//3600,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.17 func:minutes-from-duration

'<http://www.w3.org/2007/rif-builtin-function#minutes-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], _) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#minutes-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			daytimeduration(C, U, []),
			D is (integer(C)-(integer(C)//3600)*3600)//60,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.18 func:seconds-from-duration

'<http://www.w3.org/2007/rif-builtin-function#seconds-from-duration>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], _) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#seconds-from-duration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], B) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			daytimeduration(C, U, []),
			D is C-(integer(C)//60)*60,
			(	nonvar(B)
			->	D =:= B
			;	D = B
			)
		)
	).


% 4.8.1.19 func:timezone-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#timezone-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))],
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			datetime(_, _, _, _, _, _, C, U, []),
			(	ground(B)
			->	atom_codes(B, V),
				daytimeduration(D, V, []),
				D =:= C
			;	daytimeduration(C, E),
				atom_codes(B, E)
			)
		)
	).


% 4.8.1.20 func:timezone-from-date

'<http://www.w3.org/2007/rif-builtin-function#timezone-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>'))],
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			date(_, _, _, C, U, []),
			(	ground(B)
			->	atom_codes(B, V),
				daytimeduration(D, V, []),
				D =:= C
			;	daytimeduration(C, E),
				atom_codes(B, E)
			)
		)
	).


% 4.8.1.21 func:timezone-from-time

'<http://www.w3.org/2007/rif-builtin-function#timezone-from-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>'))],
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground(A)
		),
		(	atom_codes(A, U),
			time(_, _, _, C, U, []),
			(	ground(B)
			->	atom_codes(B, V),
				daytimeduration(D, V, []),
				D =:= C
			;	daytimeduration(C, E),
				atom_codes(B, E)
			)
		)
	).


% 4.8.1.22 func:subtract-dateTimes

'<http://www.w3.org/2007/rif-builtin-function#subtract-dateTimes>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			F is D-E,
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.23 func:subtract-dates

'<http://www.w3.org/2007/rif-builtin-function#subtract-dates>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			F is D-E,
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.24 func:subtract-times

'<http://www.w3.org/2007/rif-builtin-function#subtract-times>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			F is D-E,
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.25 func:add-yearMonthDurations

'<http://www.w3.org/2007/rif-builtin-function#add-yearMonthDurations>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			F is D+E,
			(	ground(C)
			->	atom_codes(C, W),
				yearmonthduration(G, W, []),
				G =:= F
			;	yearmonthduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.26 func:subtract-yearMonthDurations

'<http://www.w3.org/2007/rif-builtin-function#subtract-yearMonthDurations>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			F is D-E,
			(	ground(C)
			->	atom_codes(C, W),
				yearmonthduration(G, W, []),
				G =:= F
			;	yearmonthduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.27 func:multiply-yearMonthDuration

'<http://www.w3.org/2007/rif-builtin-function#multiply-yearMonthDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')), B],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			yearmonthduration(D, U, []),
			getnumber(B, E),
			F is integer(round(D*E-1)+1),
			(	ground(C)
			->	atom_codes(C, W),
				yearmonthduration(G, W, []),
				G =:= F
			;	yearmonthduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.28 func:divide-yearMonthDuration

'<http://www.w3.org/2007/rif-builtin-function#divide-yearMonthDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')), B],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			yearmonthduration(D, U, []),
			getnumber(B, E),
			F is integer(round(D/E-1)+1),
			(	ground(C)
			->	atom_codes(C, W),
				yearmonthduration(G, W, []),
				G =:= F
			;	yearmonthduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.29 func:divide-yearMonthDuration-by-yearMonthDuration

'<http://www.w3.org/2007/rif-builtin-function#divide-yearMonthDuration-by-yearMonthDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			F is D/E,
			(	ground(C)
			->	C =:= F
			;	C = F
			)
		)
	).


% 4.8.1.30 func:add-dayTimeDurations

'<http://www.w3.org/2007/rif-builtin-function#add-dayTimeDurations>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			F is D+E,
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.31 func:subtract-dayTimeDurations

'<http://www.w3.org/2007/rif-builtin-function#subtract-dayTimeDurations>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			F is D-E,
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.32 func:multiply-dayTimeDuration

'<http://www.w3.org/2007/rif-builtin-function#multiply-dayTimeDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')), B],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			daytimeduration(D, U, []),
			getnumber(B, E),
			F is integer(round(D*E-1)+1),
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.33 func:divide-dayTimeDuration

'<http://www.w3.org/2007/rif-builtin-function#divide-dayTimeDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')), B],
	literal(C, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			daytimeduration(D, U, []),
			getnumber(B, E),
			F is integer(round(D/E-1)+1),
			(	ground(C)
			->	atom_codes(C, W),
				daytimeduration(G, W, []),
				G =:= F
			;	daytimeduration(F, H),
				atom_codes(C, H)
			)
		)
	).


% 4.8.1.34 func:divide-dayTimeDuration-by-dayTimeDuration

'<http://www.w3.org/2007/rif-builtin-function#divide-dayTimeDuration-by-dayTimeDuration>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			F is D/E,
			(	ground(C)
			->	C =:= F
			;	C = F
			)
		)
	).


% 4.8.1.35 func:add-yearMonthDuration-to-dateTime

'<http://www.w3.org/2007/rif-builtin-function#add-yearMonthDuration-to-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, E, F, G, H, I, J, U, []),
			yearmonthduration(K, V, []),
			L is E+K-1,
			Q is D+integer(floor(L/12)),
			R is L-integer(floor(L/12))*12+1,
			memotime(datime(Q, R, F, G, H, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is M+I+31536000-N-J,
			(	ground(C)
			->	atom_codes(C, W),
				datetime(P, W, []),
				O =:= P
			;	Offset is -J,
				stamp_date_time(O, date(Year, Month, Day, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				datetime(Year, Month, Day, Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.36 func:add-yearMonthDuration-to-date

'<http://www.w3.org/2007/rif-builtin-function#add-yearMonthDuration-to-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#date>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, E, F, G, U, []),
			yearmonthduration(K, V, []),
			L is E+K-1,
			Q is D+integer(floor(L/12)),
			R is L-integer(floor(L/12))*12+1,
			memotime(datime(Q, R, F, 0, 0, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is (integer(floor(M+31536000-N-G))//60)*60,
			(	ground(C)
			->	atom_codes(C, W),
				date(P, W, []),
				O =:= P
			;	date(O, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.37 func:add-dayTimeDuration-to-dateTime

'<http://www.w3.org/2007/rif-builtin-function#add-dayTimeDuration-to-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, E, F, G, H, I, J, U, []),
			daytimeduration(K, V, []),
			L is I+K,
			memotime(datime(D, E, F, G, H, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is M+L+31536000-N-J,
			(	ground(C)
			->	atom_codes(C, W),
				datetime(P, W, []),
				O =:= P
			;	Offset is -J,
				stamp_date_time(O, date(Year, Month, Day, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				datetime(Year, Month, Day, Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.38 func:add-dayTimeDuration-to-date

'<http://www.w3.org/2007/rif-builtin-function#add-dayTimeDuration-to-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#date>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, E, F, G, U, []),
			daytimeduration(K, V, []),
			L is integer(K),
			memotime(datime(D, E, F, 0, 0, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is (integer(floor(M+L+31536000-N))//86400)*86400-G,
			(	ground(C)
			->	atom_codes(C, W),
				date(P, W, []),
				O =:= P
			;	date(O, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.39 func:add-dayTimeDuration-to-time

'<http://www.w3.org/2007/rif-builtin-function#add-dayTimeDuration-to-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#time>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, E, F, G, U, []),
			daytimeduration(K, V, []),
			L is F+K,
			memotime(datime(1972, 12, 31, D, E, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			Z is M+L+31536000-N-G,
			O is Z-86400*integer(floor(Z/86400)),
			(	ground(C)
			->	atom_codes(C, W),
				time(P, W, []),
				O =:= P-86400*integer(floor(P/86400))
			;	Offset is -G,
				stamp_date_time(O, date(_, _, _, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				time(Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.40 func:subtract-yearMonthDuration-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#subtract-yearMonthDuration-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, E, F, G, H, I, J, U, []),
			yearmonthduration(K, V, []),
			L is E-K-1,
			Q is D+integer(floor(L/12)),
			R is L-integer(floor(L/12))*12+1,
			memotime(datime(Q, R, F, G, H, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is M+I+31536000-N-J,
			(	ground(C)
			->	atom_codes(C, W),
				datetime(P, W, []),
				O =:= P
			;	Offset is -J,
				stamp_date_time(O, date(Year, Month, Day, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				datetime(Year, Month, Day, Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.41 func:subtract-yearMonthDuration-from-date

'<http://www.w3.org/2007/rif-builtin-function#subtract-yearMonthDuration-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#date>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, E, F, G, U, []),
			yearmonthduration(K, V, []),
			L is E-K-1,
			Q is D+integer(floor(L/12)),
			R is L-integer(floor(L/12))*12+1,
			memotime(datime(Q, R, F, 0, 0, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is (integer(floor(M+31536000-N-G))//60)*60,
			(	ground(C)
			->	atom_codes(C, W),
				date(P, W, []),
				O =:= P
			;	date(O, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.42 func:subtract-dayTimeDuration-from-dateTime

'<http://www.w3.org/2007/rif-builtin-function#subtract-dayTimeDuration-from-dateTime>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, E, F, G, H, I, J, U, []),
			daytimeduration(K, V, []),
			L is I-integer(K),
			memotime(datime(D, E, F, G, H, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is M+L+31536000-N-J,
			(	ground(C)
			->	atom_codes(C, W),
				datetime(P, W, []),
				O =:= P
			;	Offset is -J,
				stamp_date_time(O, date(Year, Month, Day, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				datetime(Year, Month, Day, Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.43 func:subtract-dayTimeDuration-from-date

'<http://www.w3.org/2007/rif-builtin-function#subtract-dayTimeDuration-from-date>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#date>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, E, F, G, U, []),
			daytimeduration(K, V, []),
			L is -integer(K),
			memotime(datime(D, E, F, 0, 0, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			O is (integer(floor(M+L+31536000-N))//86400)*86400-G,
			(	ground(C)
			->	atom_codes(C, W),
				date(P, W, []),
				O =:= P
			;	date(O, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.1.44 func:subtract-dayTimeDuration-from-time

'<http://www.w3.org/2007/rif-builtin-function#subtract-dayTimeDuration-from-time>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], literal(C, type('<http://www.w3.org/2001/XMLSchema#time>'))) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, E, F, G, U, []),
			daytimeduration(K, V, []),
			L is F-K,
			memotime(datime(1972, 12, 31, D, E, 0), M),
			memotime(datime(1971, 1, 1, 0, 0, 0), N),
			Z is M+L+31536000-N-G,
			O is Z-86400*integer(floor(Z/86400)),
			(	ground(C)
			->	atom_codes(C, W),
				time(P, W, []),
				O =:= P-86400*integer(floor(P/86400))
			;	Offset is -G,
				stamp_date_time(O, date(_, _, _, Hour, Minute, Second, _, _, _), Offset),
				fmsec(0, Second, Sec),
				time(Hour, Minute, Sec, Offset, S),
				atom_codes(C, S)
			)
		)
	).


% 4.8.2.1 pred:dateTime-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D =:= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.2 pred:dateTime-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-less-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D < E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.3 pred:dateTime-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-greater-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D > E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.4 pred:date-equal

'<http://www.w3.org/2007/rif-builtin-predicate#date-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D =:= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.5 pred:date-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#date-less-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D < E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.6 pred:date-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#date-greater-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D > E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.7 pred:time-equal

'<http://www.w3.org/2007/rif-builtin-predicate#time-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D =:= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.8 pred:time-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#time-less-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D < E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.9 pred:time-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#time-greater-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D > E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.10 pred:duration-equal

'<http://www.w3.org/2007/rif-builtin-predicate#duration-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#duration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#duration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			duration(D, U, []),
			duration(E, V, []),
			(	D =:= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.11 pred:dayTimeDuration-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#dayTimeDuration-less-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			(	D < E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.12 pred:dayTimeDuration-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#dayTimeDuration-greater-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			(	D > E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.13 pred:yearMonthDuration-less-than

'<http://www.w3.org/2007/rif-builtin-predicate#yearMonthDuration-less-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			(	D < E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.14 pred:yearMonthDuration-greater-than

'<http://www.w3.org/2007/rif-builtin-predicate#yearMonthDuration-greater-than>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			(	D > E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.15 pred:dateTime-not-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-not-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D =\= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.16 pred:dateTime-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-less-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D =< E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.17 pred:dateTime-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dateTime-greater-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dateTime>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			datetime(D, U, []),
			datetime(E, V, []),
			(	D >= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.18 pred:date-not-equal

'<http://www.w3.org/2007/rif-builtin-predicate#date-not-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D =\= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.19 pred:date-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#date-less-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D =< E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.20 pred:date-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#date-greater-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#date>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			date(D, U, []),
			date(E, V, []),
			(	D >= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.21 pred:time-not-equal

'<http://www.w3.org/2007/rif-builtin-predicate#time-not-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D =\= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.22 pred:time-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#time-less-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D =< E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.23 pred:time-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#time-greater-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#time>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			time(D, U, []),
			time(E, V, []),
			(	D >= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.24 pred:duration-not-equal

'<http://www.w3.org/2007/rif-builtin-predicate#duration-not-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#duration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#duration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			duration(D, U, []),
			duration(E, V, []),
			(	D =\= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.25 pred:dayTimeDuration-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dayTimeDuration-less-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			(	D =< E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.26 pred:dayTimeDuration-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#dayTimeDuration-greater-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			daytimeduration(D, U, []),
			daytimeduration(E, V, []),
			(	D >= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.27 pred:yearMonthDuration-less-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#yearMonthDuration-less-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			(	D =< E
			->	C = true
			;	C = false
			)
		)
	).


% 4.8.2.28 pred:yearMonthDuration-greater-than-or-equal

'<http://www.w3.org/2007/rif-builtin-predicate#yearMonthDuration-greater-than-or-equal>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'))], C) :-
	when(
		(	ground([A, B])
		),
		(	atom_codes(A, U),
			atom_codes(B, V),
			yearmonthduration(D, U, []),
			yearmonthduration(E, V, []),
			(	D >= E
			->	C = true
			;	C = false
			)
		)
	).


% 4.10.1.1 func:PlainLiteral-from-string-lang

'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-from-string-lang>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))],
	literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-from-string-lang>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')),
	literal(B, type('<http://www.w3.org/2001/XMLSchema#string>'))], literal(A, lang(C))) :-
	downcase_atom(B, C).


% 4.10.1.2 func:string-from-PlainLiteral

'<http://www.w3.org/2007/rif-builtin-function#string-from-PlainLiteral>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))], literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#string-from-PlainLiteral>'([literal(A, lang(_))], literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))).


% 4.10.1.3 func:lang-from-PlainLiteral

'<http://www.w3.org/2007/rif-builtin-function#lang-from-PlainLiteral>'([literal(_, type('<http://www.w3.org/2001/XMLSchema#string>'))], literal('', type('<http://www.w3.org/2001/XMLSchema#string>'))) :-
	!.
'<http://www.w3.org/2007/rif-builtin-function#lang-from-PlainLiteral>'([literal(_, lang(A))], literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))).


% 4.10.1.4 func:PlainLiteral-compare @@partial implementation: no collation

'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-compare>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))], D) :-
	!,
	(	A @< C
	->	D = -1
	;	(	A == C
		->	D = 0
		;	(	A @> C
			->	D = 1
			)
		)
	).
'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-compare>'([literal(A, lang(B)), literal(C, lang(B))], D) :-
	(	A @< C
	->	D = -1
	;	(	A == C
		->	D = 0
		;	(	A @> C
			->	D = 1
			)
		)
	).


% 4.10.1.5 func:PlainLiteral-length

'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-length>'([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))], C) :-
	!,
	sub_atom(A, 0, C, 0, _).
'<http://www.w3.org/2007/rif-builtin-function#PlainLiteral-length>'([literal(A, lang(_))], C) :-
	sub_atom(A, 0, C, 0, _).


% 4.10.2.1 pred:matches-language-range @@partial implementation: no false results

'<http://www.w3.org/2007/rif-builtin-predicate#matches-language-range>'([literal(A, lang(B)), literal(C, type('<http://www.w3.org/2001/XMLSchema#string>'))], true) :-
	A \= '',
	atom_codes(C, U),
	regexp_wildcard(U, V),
	atom_codes(E, V),
	atomic_list_concat(['^', E], F),
	downcase_atom(F, G),
	downcase_atom(B, H),
	atom_codes(G, I),
	atom_codes(H, J),
	regex(I, J, _).


% 4.11.3.1 pred:is-list

'<http://www.w3.org/2007/rif-builtin-predicate#is-list>'([A], B) :-
	(	is_list(A)
	->	B = true
	;	B = false
	).


% 4.11.3.2 pred:list-contains

'<http://www.w3.org/2007/rif-builtin-predicate#list-contains>'([A, B], C) :-
	when(
		(	nonvar(A)
		),
		(	member(B, A)
		->	C = true
		;	C = false
		)
	).


% 4.11.4.1 func:make-list

'<http://www.w3.org/2007/rif-builtin-function#make-list>'(A, A).


% 4.11.4.2 func:count

'<http://www.w3.org/2007/rif-builtin-function#count>'([A], B) :-
	when(
		(	nonvar(A)
		),
		(	length(A, B)
		)
	).


% 4.11.4.3 func:get

'<http://www.w3.org/2007/rif-builtin-function#get>'([A, B], C) :-
	when(
		(	nonvar(A),
			ground(B)
		),
		(	getnumber(B, U),
			nth0(U, A, C)
		)
	).


% 4.11.4.4 func:sublist

'<http://www.w3.org/2007/rif-builtin-function#sublist>'([A, B, C], D) :-
	!,
	when(
		(	nonvar(A),
			ground([B, C])
		),
		(	getint(B, U),
			getint(C, V),
			length(A, W),
			(	U < 0
			->	I is W+U
			;	I is U
			),
			(	V < 0
			->	J is W+V
			;	J is V
			),
			append(E, F, A),
			length(E, I),
			append(D, G, F),
			K is J-I,
			(	length(D, K)
			->	true
			;	G = []
			),
			!
		)
	).
'<http://www.w3.org/2007/rif-builtin-function#sublist>'([A, B], C) :-
	when(
		(	nonvar(A),
			ground(B)
		),
		(	getint(B, U),
			length(A, W),
			(	U < 0
			->	I is W+U
			;	I is U
			),
			append(E, C, A),
			length(E, I),
			!
		)
	).


% 4.11.4.5 func:append

'<http://www.w3.org/2007/rif-builtin-function#append>'([A|B], C) :-
	append(A, B, C).


% 4.11.4.6 func:concatenate

'<http://www.w3.org/2007/rif-builtin-function#concatenate>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	append(A, B)
		)
	).


% 4.11.4.7 func:insert-before

'<http://www.w3.org/2007/rif-builtin-function#insert-before>'([A, B, C], D) :-
	when(
		(	nonvar(A),
			ground([B, C])
		),
		(	getint(B, U),
			length(A, W),
			(	U < 0
			->	I is W+U
			;	I is U
			),
			append(G, H, A),
			length(G, I),
			append([G, [C], H], D)
		)
	).


% 4.11.4.8 func:remove

'<http://www.w3.org/2007/rif-builtin-function#remove>'([A, B], C) :-
	when(
		(	nonvar(A),
			ground(B)
		),
		(	getint(B, U),
			length(A, W),
			(	U < 0
			->	I is W+U
			;	I is U
			),
			append(G, [_|T], A),
			length(G, I),
			append(G, T, C)
		)
	).


% 4.11.4.9 func:reverse

'<http://www.w3.org/2007/rif-builtin-function#reverse>'([A], B) :-
	reverse(A, B).


% 4.11.4.10 func:index-of

'<http://www.w3.org/2007/rif-builtin-function#index-of>'([A, B], C) :-
	when(
		(	nonvar(A),
			ground(B)
		),
		(	findall(I,
				(	nth0(I, A, B)
				),
				C
			)
		)
	).


% 4.11.4.11 func:union

'<http://www.w3.org/2007/rif-builtin-function#union>'(A, B) :-
	when(
		(	nonvar(A)
		),
		(	append(A, C),
			distinct(C, B)
		)
	).


% 4.11.4.12 func:distinct-values

'<http://www.w3.org/2007/rif-builtin-function#distinct-values>'([A], B) :-
	when(
		(	nonvar(A)
		),
		(	distinct(A, B)
		)
	).


% 4.11.4.13 func:intersect

'<http://www.w3.org/2007/rif-builtin-function#intersect>'([A, B], C) :-
	when(
		(	ground(A),
			ground(B)
		),
		(	findall(I,
				(	member(I, A),
					member(I, B)
				),
				C
			)
		)
	).


% 4.11.4.14 func:except

'<http://www.w3.org/2007/rif-builtin-function#except>'([A, B], C) :-
	when(
		(	ground(A),
			ground(B)
		),
		(	findall(I,
				(	member(I, A),
					\+member(I, B)
				),
				C
			)
		)
	).


% Prolog built-ins

prolog_sym(abolish, abolish, rel).
prolog_sym(abort, abort, rel).
prolog_sym(abs, abs, func).
prolog_sym(absolute_file_name, absolute_file_name, rel).
prolog_sym(acos, acos, func).
prolog_sym(acosh, acosh, func).
prolog_sym(acyclic_term, acyclic_term, rel).
prolog_sym(alarm, alarm, rel).
prolog_sym(append, append, rel).
prolog_sym(arg, arg, rel).
prolog_sym(arithmetic_equal, =:=, rel).
prolog_sym(arithmetic_greater_than, >, rel).
prolog_sym(arithmetic_greater_than_or_equal, >=, rel).
prolog_sym(arithmetic_less_than, <, rel).
prolog_sym(arithmetic_less_than_or_equal, =<, rel).
prolog_sym(arithmetic_not_equal, =\=, rel).
prolog_sym(asin, asin, func).
prolog_sym(asinh, asinh, func).
prolog_sym(assert, assert, rel).
prolog_sym(asserta, asserta, rel).
prolog_sym(assertz, assertz, rel).
prolog_sym(at_end_of_stream, at_end_of_stream, rel).
prolog_sym(atan, atan, func).
prolog_sym(atan2, atan2, func).
prolog_sym(atanh, atanh, func).
prolog_sym(atom, atom, rel).
prolog_sym(atom_chars, atom_chars, rel).
prolog_sym(atom_codes, atom_codes, rel).
prolog_sym(atom_concat, atom_concat, rel).
prolog_sym(atom_length, atom_length, rel).
prolog_sym(atom_number, atom_number, rel).
prolog_sym(atomic, atomic, rel).
prolog_sym(atomic_concat, atomic_concat, rel).
prolog_sym(atomic_list_concat, atomic_list_concat, rel).
prolog_sym(b_getval, b_getval, rel).
prolog_sym(b_setval, b_setval, rel).
prolog_sym(bagof, bagof, rel).
prolog_sym(between, between, rel).
prolog_sym(break, break, rel).
prolog_sym(call, call, rel).
prolog_sym(call_residue_vars, call_residue_vars, rel).
prolog_sym(callable, callable, rel).
prolog_sym(catch, catch, rel).
prolog_sym(ceiling, ceiling, func).
prolog_sym(char_code, char_code, rel).
prolog_sym(char_conversion, char_conversion, rel).
prolog_sym(char_type, char_type, rel).
prolog_sym(character_count, character_count, rel).
prolog_sym(clause, clause, rel).
prolog_sym(close, close, rel).
prolog_sym(code_type, code_type, rel).
prolog_sym(compare, compare, rel).
prolog_sym(compound, compound, rel).
prolog_sym(conjunction, ',', rel).
prolog_sym(copy_term, copy_term, rel).
prolog_sym(copy_term_nat, copy_term_nat, rel).
prolog_sym(cos, cos, func).
prolog_sym(cosh, cosh, func).
prolog_sym(cputime, cputime, func).
prolog_sym(create_mutable, create_mutable, rel).
prolog_sym(create_prolog_flag, create_prolog_flag, rel).
prolog_sym(current_atom, current_atom, rel).
prolog_sym(current_char_conversion, current_char_conversion, rel).
prolog_sym(current_input, current_input, rel).
prolog_sym(current_key, current_key, rel).
prolog_sym(current_module, current_module, rel).
prolog_sym(current_op, current_op, rel).
prolog_sym(current_output, current_output, rel).
prolog_sym(current_predicate, current_predicate, rel).
prolog_sym(current_prolog_flag, current_prolog_flag, rel).
prolog_sym(cut, !, rel).
prolog_sym(cyclic_term, cyclic_term, rel).
prolog_sym(date_time_stamp, date_time_stamp, rel).
prolog_sym(date_time_value, date_time_value, rel).
prolog_sym(day_of_the_week, day_of_the_week, rel).
prolog_sym(delete, delete, rel).
prolog_sym(dif, dif, rel).
prolog_sym(discontiguous, discontiguous, rel).
prolog_sym(disjunction, ;, rel).
prolog_sym(display, display, rel).
prolog_sym(div, div, func).
prolog_sym(duplicate_term, duplicate_term, rel).
prolog_sym(dynamic, dynamic, rel).
prolog_sym(e, e, func).
prolog_sym(ensure_loaded, ensure_loaded, rel).
prolog_sym(environ, environ, rel).
prolog_sym(epsilon, epsilon, func).
prolog_sym(erase, erase, rel).
prolog_sym(erf, erf, func).
prolog_sym(erfc, erfc, func).
prolog_sym(exception, exception, rel).
prolog_sym(exists, exists, rel).
prolog_sym(exp, exp, func).
prolog_sym(fail, fail, rel).
prolog_sym(false, false, rel).
prolog_sym(file_base_name, file_base_name, rel).
prolog_sym(file_name_extension, file_name_extension, rel).
prolog_sym(findall, findall, rel).
prolog_sym(flatten, flatten, rel).
prolog_sym(float, float, rel).
prolog_sym(float_fractional_part, float_fractional_part, func).
prolog_sym(float_function, float, func).
prolog_sym(float_integer_part, float_integer_part, func).
prolog_sym(floor, floor, func).
prolog_sym(flush_output, flush_output, rel).
prolog_sym(forall, forall, rel).
prolog_sym(format, format, rel).
prolog_sym(format_time, format_time, rel).
prolog_sym(freeze, freeze, rel).
prolog_sym(frozen, frozen, rel).
prolog_sym(functor, functor, rel).
prolog_sym(garbage_collect, garbage_collect, rel).
prolog_sym(garbage_collect_atoms, garbage_collect_atoms, rel).
prolog_sym(gc, gc, rel).
prolog_sym(gcd, gcd, func).
prolog_sym(get, get, rel).
prolog_sym(get_byte, get_byte, rel).
prolog_sym(get_char, get_char, rel).
prolog_sym(get_code, get_code, rel).
prolog_sym(get_mutable, get_mutable, rel).
prolog_sym(get_time, get_time, rel).
prolog_sym(get0, get0, rel).
prolog_sym(getcwd, getcwd, rel).
prolog_sym(ground, ground, rel).
prolog_sym(halt, halt, rel).
prolog_sym(if, soft_cut, rel).
prolog_sym(if_then, ->, rel).
prolog_sym(if_then_else, if_then_else, rel).
prolog_sym(ignore, ignore, rel).
prolog_sym(include, include, rel).
prolog_sym(initialization, initialization, rel).
prolog_sym(instance, instance, rel).
prolog_sym(integer, integer, rel).
prolog_sym(integer_conjunction, /\, func).
prolog_sym(integer_disjunction, \/, func).
prolog_sym(integer_exclusive_disjunction, xor, func).
prolog_sym(integer_function, integer, func).
prolog_sym(integer_left_logical_shift, <<, func).
prolog_sym(integer_negation, \, func).
prolog_sym(integer_power, ^, func).
prolog_sym(integer_quotient, //, func).
prolog_sym(integer_right_logical_shift, >>, func).
prolog_sym(is, is, rel).
prolog_sym(is_list, is_list, rel).
prolog_sym(is_stream, is_stream, rel).
prolog_sym(keysort, keysort, rel).
prolog_sym(last, last, rel).
prolog_sym(length, length, rel).
prolog_sym(lgamma, lgamma, func).
prolog_sym(line_count, line_count, rel).
prolog_sym(line_position, line_position, rel).
prolog_sym(listing, listing, rel).
prolog_sym(log, log, func).
prolog_sym(log10, log10, func).
prolog_sym(lsb, lsb, func).
prolog_sym(max, max, func).
prolog_sym(max_list, max_list, rel).
prolog_sym(member, member, rel).
prolog_sym(memberchk, memberchk, rel).
prolog_sym(message_to_string, message_to_string, rel).
prolog_sym(min, min, func).
prolog_sym(min_list, min_list, rel).
prolog_sym(minus, -, func).
prolog_sym(mod, mod, func).
prolog_sym(msb, msb, func).
prolog_sym(multifile, multifile, rel).
prolog_sym(name, name, rel).
prolog_sym(nb_current, nb_current, rel).
prolog_sym(nb_delete, nb_delete, rel).
prolog_sym(nb_getval, nb_getval, rel).
prolog_sym(nb_linkarg, nb_linkarg, rel).
prolog_sym(nb_linkval, nb_linkval, rel).
prolog_sym(nb_setarg, nb_setarg, rel).
prolog_sym(nb_setval, nb_setval, rel).
prolog_sym(nl, nl, rel).
prolog_sym(nonvar, nonvar, rel).
prolog_sym(not_provable, \+, rel).
prolog_sym(not_unifiable, \=, rel).
prolog_sym(nth, nth, rel).
prolog_sym(nth_clause, nth_clause, rel).
prolog_sym(nth0, nth0, rel).
prolog_sym(nth1, nth1, rel).
prolog_sym(number, number, rel).
prolog_sym(number_chars, number_chars, rel).
prolog_sym(number_codes, number_codes, rel).
prolog_sym(numbervars, numbervars, rel).
prolog_sym(numlist, numlist, rel).
prolog_sym(on_signal, on_signal, rel).
prolog_sym(once, once, rel).
prolog_sym(op, op, rel).
prolog_sym(open, open, rel).
prolog_sym(parse_time, parse_time, rel).
prolog_sym(peek_byte, peek_byte, rel).
prolog_sym(peek_char, peek_char, rel).
prolog_sym(peek_code, peek_code, rel).
prolog_sym(permutation, permutation, rel).
prolog_sym(pi, pi, func).
prolog_sym(plus, +, rel).
prolog_sym(plus_function, +, func).
prolog_sym(popcount, popcount, func).
prolog_sym(portray_clause, portray_clause, rel).
prolog_sym(power, **, func).
prolog_sym(predicate_property, predicate_property, rel).
prolog_sym(predsort, predsort, rel).
prolog_sym(print, print, rel).
prolog_sym(print_message, print_message, rel).
prolog_sym(print_message_lines, print_message_lines, rel).
prolog_sym(product, *, func).
prolog_sym(prolog_flag, prolog_flag, rel).
prolog_sym(prolog_load_context, prolog_load_context, rel).
prolog_sym(prompt, prompt, rel).
prolog_sym(put, put, rel).
prolog_sym(put_byte, put_byte, rel).
prolog_sym(put_char, put_char, rel).
prolog_sym(put_code, put_code, rel).
prolog_sym(quotient, /, func).
prolog_sym(random, random, func).
prolog_sym(rational, rational, rel).
prolog_sym(rational_function, rational, func).
prolog_sym(rationalize, rationalize, func).
prolog_sym(read, read, rel).
prolog_sym(read_term, read_term, rel).
prolog_sym(recorda, recorda, rel).
prolog_sym(recorded, recorded, rel).
prolog_sym(recordz, recordz, rel).
prolog_sym(rem, rem, func).
prolog_sym(rename_file, rename_file, rel).
prolog_sym(repeat, repeat, rel).
prolog_sym(retract, retract, rel).
prolog_sym(retractall, retractall, rel).
prolog_sym(reverse, reverse, rel).
prolog_sym(round, round, func).
prolog_sym(same_length, same_length, rel).
prolog_sym(see, see, rel).
prolog_sym(seeing, seeing, rel).
prolog_sym(seen, seen, rel).
prolog_sym(select, select, rel).
prolog_sym(selectchk, selectchk, rel).
prolog_sym(set_input, set_input, rel).
prolog_sym(set_output, set_output, rel).
prolog_sym(set_prolog_flag, set_prolog_flag, rel).
prolog_sym(set_stream_position, set_stream_position, rel).
prolog_sym(setarg, setarg, rel).
prolog_sym(setof, setof, rel).
prolog_sym(set_random, set_random, rel).
prolog_sym(sign, sign, func).
prolog_sym(simple, simple, rel).
prolog_sym(sin, sin, func).
prolog_sym(sinh, sinh, func).
prolog_sym(skip, skip, rel).
prolog_sym(sort, sort, rel).
prolog_sym(source_file, source_file, rel).
prolog_sym(source_location, source_location, rel).
prolog_sym(sqrt, sqrt, func).
prolog_sym(stamp_date_time, stamp_date_time, rel).
prolog_sym(statistics, statistics, rel).
prolog_sym(stream_position, stream_position, rel).
prolog_sym(stream_position_data, stream_position_data, rel).
prolog_sym(stream_property, stream_property, rel).
prolog_sym(sub_atom, sub_atom, rel).
prolog_sym(sublist, sublist, rel).
prolog_sym(subsumes_term, subsumes_term, rel).
prolog_sym(succ, succ, rel).
prolog_sym(sum_list, sum_list, rel).
prolog_sym(tab, tab, rel).
prolog_sym(tan, tan, func).
prolog_sym(tanh, tanh, func).
prolog_sym(tell, tell, rel).
prolog_sym(telling, telling, rel).
prolog_sym(term_greater_than, @>, rel).
prolog_sym(term_greater_than_or_equal, @>=, rel).
prolog_sym(term_hash, term_index, rel).
prolog_sym(term_identical, ==, rel).
prolog_sym(term_less_than, @<, rel).
prolog_sym(term_less_than_or_equal, @=<, rel).
prolog_sym(term_not_identical, \==, rel).
prolog_sym(term_to_atom, term_to_atom, rel).
prolog_sym(term_variables, term_variables, rel).
prolog_sym(throw, throw, rel).
prolog_sym(time, time, rel).
prolog_sym(time_file, time_file, rel).
prolog_sym(told, told, rel).
prolog_sym(true, true, rel).
prolog_sym(truncate, truncate, func).
prolog_sym(unifiable, unifiable, rel).
prolog_sym(unify, =, rel).
prolog_sym(unify_with_occurs_check, unify_with_occurs_check, rel).
prolog_sym(univ, =.., rel).
prolog_sym(unknown, unknown, rel).
prolog_sym(update_mutable, update_mutable, rel).
prolog_sym(var, var, rel).
prolog_sym(variant, variant, rel).
prolog_sym(version, version, rel).
prolog_sym(when, when, rel).
prolog_sym(with_output_to, with_output_to, rel).
prolog_sym(write, write, rel).
prolog_sym(write_canonical, write_canonical, rel).
prolog_sym(write_term, write_term, rel).
prolog_sym(writeln, writeln, rel).
prolog_sym(writeq, writeq, rel).


% support

def_pfx('math:', '<http://www.w3.org/2000/10/swap/math#>').
def_pfx('e:', '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#>').
def_pfx('list:', '<http://www.w3.org/2000/10/swap/list#>').
def_pfx('xsd:', '<http://www.w3.org/2001/XMLSchema#>').
def_pfx('log:', '<http://www.w3.org/2000/10/swap/log#>').
def_pfx('r:', '<http://www.w3.org/2000/10/swap/reason#>').
def_pfx('rdfs:', '<http://www.w3.org/2000/01/rdf-schema#>').
def_pfx('time:', '<http://www.w3.org/2000/10/swap/time#>').
def_pfx('rdf:', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#>').
def_pfx('string:', '<http://www.w3.org/2000/10/swap/string#>').
def_pfx('owl:', '<http://www.w3.org/2002/07/owl#>').
def_pfx('n3:', '<http://www.w3.org/2004/06/rei#>').


put_pfx(_, URI) :-
	atomic_list_concat(['<', URI, '>'], U),
	pfx(_, U),
	!.
put_pfx(_, URI) :-
	atomic_list_concat(['<', URI, '>'], U),
	def_pfx(Pf, U),
	\+pfx(Pf, _),
	!,
	assertz(pfx(Pf, U)).
put_pfx(Pf, URI) :-
	atomic_list_concat(['<', URI, '>'], U),
	fresh_pf(Pf, Pff),
	assertz(pfx(Pff, U)).


fresh_pf(Pf, Pfx) :-
	atom_concat(Pf, ':', Pfx),
	\+pfx(Pfx, _),
	!.
fresh_pf(_, Pfx) :-
	gensym(ns, Pfn),
	fresh_pf(Pfn, Pfx).


cnt(A) :-
	nb_getval(A, B),
	C is B+1,
	nb_setval(A, C),
	(	flag('debug-cnt'),
		C mod 1000 =:= 0
	->	format(user_error, '~w = ~w~n', [A, C]),
		flush_output(user_error)
	;	true
	).


cnt(A, I) :-
	nb_getval(A, B),
	C is B+I,
	nb_setval(A, C),
	(	flag('debug-cnt'),
		C mod 1000 =:= 0
	->	format(user_error, '~w = ~w~n', [A, C]),
		flush_output(user_error)
	;	true
	).


within_scope([A, B]) :-
	(	var(B)
	->	B = 1
	;	true
	),
	(	(	flag('no-span')	% DEPRECATED
		;	B = 0
		)
	->	brake
	;	nb_getval(limit, C),
		(	C < B
		->	nb_setval(limit, B)
		;	true
		),
		span(B)
	),
	nb_getval(scope, A).


exopred(P, S, O) :-
	(	var(P)
	->	pred(P)
	;	atom(P),
		current_predicate(P/2)
	),
	call(P, S, O).


unify(A, B) :-
	nonvar(A),
	A = exopred(P, S, O),
	unify(S, T),
	unify(O, R),
	(	(	nonvar(B)
		;	nonvar(P)
		)
	->	(	nonvar(P)
		->	atom(P)
		;	true
		),
		B =.. [P, T, R],
		atom(P)
	),
	!.
unify(A, B) :-
	nonvar(B),
	B = exopred(P, S, O),
	unify(S, T),
	unify(O, R),
	(	(	nonvar(A)
		;	nonvar(P)
		)
	->	(	nonvar(P)
		->	atom(P)
		;	true
		),
		A =.. [P, T, R],
		atom(P)
	),
	!.
unify(A, B) :-
	nonvar(A),
	nonvar(B),
	A = cn(_),
	B = cn(_),
	!,
	(	ground(A)
	->	'<http://www.w3.org/2000/10/swap/log#includes>'(A, B),
		'<http://www.w3.org/2000/10/swap/log#includes>'(B, A)
	;	'<http://www.w3.org/2000/10/swap/log#includes>'(B, A),
		'<http://www.w3.org/2000/10/swap/log#includes>'(A, B)
	).
unify(A, B) :-
	nonvar(A),
	nonvar(B),
	A =.. [P, S, O],
	B =.. [P, T, R],
	!,
	unify(S, T),
	unify(O, R).
unify(A, A).


cn([A|B]) :-
	call(A),
	(	B = [C]
	->	true
	;	C = cn(B)
	),
	call(C).


clist([], true) :-
	!.
clist([], pass(_)) :-
	!.
clist([A], A) :-
	A \= cn(_),
	!.
clist(A, cn(A)).


clistflat([], true) :-
	!.
clistflat([A], A) :-
	A \= cn(_),
	!.
clistflat(A, cn(B)) :-
	(	nonvar(A)
	->	cflat(A, C),
		distinct(C, B)
	;	distinct(B, A)
	).


cflat([], []) :-
	!.
cflat([A|B], C) :-
	cflat(B, D),
	copy_term_nat(A, E),
	(	E = cn(F)
	->	append(F, D, C)
	;	(	E = true
		->	C = D
		;	C = [E|D]
		)
	).


cmember(A, cn(B)) :-
	member(A, B).
cmember(A, A) :-
	A \= cn(_).


clast(cn(A), B) :-
	!,
	last(A, B).
clast(A, A).


cn_conj(A, B) :-
	clist(C, A),
	c_d(C, D),
	c_list(D, B).


conj_cn(A, B) :-
	c_list(C, A),
	c_d(C, D),
	clist(D, B).


c_d([], []) :-
	!.
c_d([(A;B)|C], [(D;E)|F]) :-
	!,
	cn_conj(A, D),
	cn_conj(B, E),
	c_d(C, F).
c_d([A|B], [A|C]) :-
	c_d(B, C).


c_list([], true) :-
	!.
c_list([], pass(_)) :-
	!.
c_list([A], A) :-
	A \= (_, _),
	!.
c_list([A|B], (A, C)) :-
	c_list(B, C).


c_append((A, B), C, (A, D)) :-
	c_append(B, C, D),
	!.
c_append(A, B, (A, B)).


couple([], [], []).
couple([A|B], [C|D], [[A, C]|E]) :-
	couple(B, D, E).


agraph(N, cn([X|Y])) :-
	!,
	unify(X, U),
	(	\+graph(N, U)
	->	assertz(graph(N, U))
	;	true
	),
	(	Y = [Z]
	->	true
	;	Z = cn(Y)
	),
	agraph(N, Z).
agraph(N, X) :-
	unify(X, U),
	(	\+graph(N, U)
	->	assertz(graph(N, U))
	;	true
	).


qgraph(N, cn([X|Y])) :-
	!,
	(	X = exopred(_, _, _)
	->	graph(N, T),
		unify(T, X)
	;	graph(N, X)
	),
	(	Y = [Z]
	->	true
	;	Z = cn(Y)
	),
	qgraph(N, Z).
qgraph(N, X) :-
	(	X = exopred(_, _, _)
	->	graph(N, T),
		unify(T, X)
	;	graph(N, X)
	).


conjoin([X], X) :-
	!.
conjoin([true|Y], Z) :-
	conjoin(Y, Z),
	!.
conjoin([X|Y], Z) :-
	conjoin(Y, C),
	clist(U, X),
	clist(V, C),
	conjoin(U, V, W),
	sort(W, D),
	clist(D, Z).


conjoin([], U, U) :-
	!.
conjoin([X|Y], U, V) :-
	member(Z, U),
	unify(X, Z),	
	!,
	conjoin(Y, U, V).
conjoin([X|Y], U, [X|V]) :-
	conjoin(Y, U, V).


difference([true, _], true) :-
	!.
difference([X, true], X) :-
	!.
difference([X, Y], Z) :-
	clist(U, X),
	clist(V, Y),
	difference(U, V, W),
	clist(W, Z).

difference([], _, []) :-
	!.
difference([X|Y], U, V) :-
	member(Z, U),
	unify(X, Z),
	!,
	difference(Y, U, V).
difference([X|Y], U, [X|V]) :-
	difference(Y, U, V).


intersect([X], X) :-
	!.
intersect([true|_], true) :-
	!.
intersect([X|Y], Z) :-
	intersect(Y, I),
	clist(U, X),
	clist(V, I),
	intersect(U, V, W),
	clist(W, Z).


intersect([], _, []) :-
	!.
intersect([X|Y], U, V) :-
	member(Z, U),
	unify(X, Z),
	!,
	V = [X|W],
	intersect(Y, U, W).
intersect([_|Y], U, V) :-
	intersect(Y, U, V).


cartesian([], []).
cartesian([A|B], [C|D]) :-
	member(C, A),
	cartesian(B, D).


distinct(A, B) :-
	(	ground(A)
	->	distinct_hash(A, B)
	;	distinct_value(A, B)
	).


distinct_hash([], []) :-
	(	retract(hash_value(_, _)),
		fail
	;	true
	),
	!.
distinct_hash([A|B], C) :-
	term_index(A, D),
	(	hash_value(D, E)
	->	(	unify(A, E)
		->	C = F
		;	C = [A|F]
		)
	;	assertz(hash_value(D, A)),
		C = [A|F]
	),
	distinct_hash(B, F).


distinct_value([], []).
distinct_value([A|B], [A|D]) :-
	del(B, A, E),
	distinct_value(E, D).


del([], _, []).
del([A|B], C, D) :-
	copy_term_nat(A, Ac),
	copy_term_nat(C, Cc),
	unify(Ac, Cc),
	!,
	del(B, C, D).
del([A|B], C, [A|D]) :-
	del(B, C, D).


subst(_, [], []).
subst(A, B, C) :-
	member([D, E], A),
	append(D, F, B),
	!,
	append(E, G, C),
	subst(A, F, G).
subst(A, [B|C], [B|D]) :-
	subst(A, C, D).


quicksort([], []).
quicksort([A|B], C) :-
	split(A, B, D, E),
	quicksort(D, F),
	quicksort(E, G),
	append(F, [A|G], C).


split(_, [], [], []).
split(A, [B|C], [B|D], E) :-
	sort([A, B], [B, A]),
	!,
	split(A, C, D, E).
split(A, [B|C], D, [B|E]) :-
	split(A, C, D, E).


last_tail([], []) :-
	!.
last_tail([_|B], B) :-
	\+is_list(B),
	!.
last_tail([_|B], C) :-
	last_tail(B, C).


sub_list(A, A) :-
	!.
sub_list([A|B], C) :-
	sub_list(B, A, C).


sub_list(A, _, A) :-
	!.
sub_list([A|B], C, [C|D]) :-
	!,
	sub_list(B, A, D).
sub_list([A|B], _, C) :-
	sub_list(B, A, C).


sum([], 0) :-
	!.
sum([A|B], C) :-
	getnumber(A, X),
	sum(B, D),
	C is X+D.


product([], 1) :-
	!.
product([A|B], C) :-
	getnumber(A, X),
	product(B, D),
	C is X*D.


rms(A, B) :-
	findall(C,
		(	member(D, A),
			getnumber(D, E),
			C is E*E
		),
		F
	),
	sum(F, G),
	length(F, H),
	B is sqrt(G/H).


bmax([A|B], C) :-
	bmax(B, A, C).


bmax([], A, A).
bmax([A|B], C, D) :-
	getnumber(A, X),
	getnumber(C, Y),
	(	X > Y
	->	bmax(B, A, D)
	;	bmax(B, C, D)
	).


bmin([A|B], C) :-
	bmin(B, A, C).


bmin([], A, A).
bmin([A|B], C, D) :-
	getnumber(A, X),
	getnumber(C, Y),
	(	X < Y
	->	bmin(B, A, D)
	;	bmin(B, C, D)
	).


inconsistent(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>')|B]) :-
	memberchk('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>'), B),
	!.
inconsistent(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>')|B]) :-
	memberchk('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>'), B),
	!.
inconsistent([_|B]) :-
	inconsistent(B).


inverse('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>'),
	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>')) :-
	!.
inverse('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>'),
	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>')).


bnet :-
	(	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>'([A|B], _),
		sort(B, C),
		findall(Y,
			(	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#conditional>'([A|X], Y),
				sort(X, C)
			),
			L
		),
		sum(L, S),
		length(L, N),
		Z is S/N,
		\+bcnd([A|B], _),
		assertz(bcnd([A|B], Z)),
		inverse(A, D),
		\+bcnd([D|B], _),
		E is 1-Z,
		assertz(bcnd([D|B], E)),
		fail
	;	bcnd(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, _)|B], _),
		(	\+bvar(A),
			assertz(bvar(A))
		;	true
		),
		member('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(C, _), B),
		\+bref(C, A),
		assertz(bref(C, A)),
		\+bvar(C),
		assertz(bvar(C)),
		fail
	;	true
	).


bval('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>').
bval('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>').


brel('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, _), '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(B, _)) :-
	bref(A, B),
	!.
brel(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(B, _)) :-
	bref(C, B),
	brel(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(C, _)).


bpar([], []) :-
	!.
bpar(['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, _)|B], [A|C]) :-
	bpar(B, C).


bget(A, B, 1.0) :-
	memberchk(A, B),
	!.
bget('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>'), B, 0.0) :-
	memberchk('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>'), B),
	!.
bget('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#F>'), B, C) :-
	(	memberchk('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>'), B),
		!,
		C is 0.0
	;
		!,
		bget('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#boolean>'(A, '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#T>'), B, D),
		C is 1-D
	).
bget(A, B, C) :-
	(	bgot(A, B, C)
	->	true
	;	(	member(X, B),
			brel(A, X),
			member(G, B),
			findall(_,
				(	member(Z, [A|B]),
					brel(G, Z)
				),
				[]
			),
			del(B, G, H),
			!,
			bget(G, [A|H], U),
			bget(A, H, V),
			bget(G, H, W),
			(	W < 1e-15
			->	C is 0.5
			;	E is U*V/W,
				bmin([E, 1.0], C)
			)
		;	findall([Z, Y],
				(	bcnd([A|O], P),
					bcon(O, B, Q),
					Z is P*Q,
					bpar(O, Y)
				),
				L
			),
			findall(Z,
				(	member([_, Z], L)
				),
				N
			),
			distinct(N, I),
			findall(Z,
				(	member(Y, I),
					findall(P,
						(	member([P, Y], L)
						),
						Q
					),
					sum(Q, R),
					length(Q, S),
					length(Y, T),
					(	Q = []
					->	Z is 0.0
					;	D is 2**(T-ceiling(log(S)/log(2))),
						(	D < 1
						->	Z is R*D
						;	Z is R
						)
					)
				),
				J
			),
			(	J = []
			->	C is 0.0
			;	bmax(J, C)
			)
		),
		assertz(bgot(A, B, C))
	).


bcon([], _, 1.0) :-
	!.
bcon(_, B, 0.5) :-
	inconsistent(B),
	!.
bcon([A|B], C, D) :-
	bget(A, C, E),
	bcon(B, [A|C], F),
	D is E*F.


tmp_file(A) :-
	(	current_prolog_flag(dialect, swi),
		current_prolog_flag(windows, true),
		current_prolog_flag(pid, B)
	->	atomic_list_concat(['pl_eye_', B, '_'], C)
	;	C = 'eye'
	),
	tmp_file(C, A).


readln(In, Line):-
	get_code(In, Code),
	readln(Code, Codes, In),
	atom_codes(Line, Codes).


readln(10, [], _) :-
	!.
readln(-1, [], _) :-
	!.
readln(Code, [Code|Codes], In) :-
	get_code(In, Next),
	readln(Next, Codes, In).


:- if(current_predicate(operating_system_support:system/2)).
exec(A, B) :-
	(	system(A, B)
	->	true
	;	B = 1
	),
	(	B =:= 0
	->	true
	;	throw(exec_error(A))
	).
:- else.
exec(A, B) :-
	shell(A, B),
	(	B =:= 0
	->	true
	;	throw(exec_error(A))
	).
:- endif.


:- if(\+current_predicate(getcwd/1)).
getcwd(A) :-
	working_directory(A, A).
:- endif.


% Modified Base64 for XML identifiers

base64xml(A, B) :-
	base64(A, C),
	atom_codes(C, D),
	subst([[[0'+], [0'_]], [[0'/], [0':]], [[0'=], []]], D, E),
	atom_codes(B, E).


:- if(current_prolog_flag(dialect, swi)).
term_index(A, B) :-
	term_hash(A, B).
:- else.
term_index(A, B) :-
	(	ground(A)
	->	term_hash(A, B)
	;	true
	).
:- endif.


term_arg_1(A, B) :-
	compound(A),
	!,
	arg(1, A, C),
	term_index(C, B).
term_arg_1(_, void).


if_then_else(A, B, C) :-
	(	call(A)
	->	call(B)
	;	call(C)
	).


soft_cut(A, B, C) :-
	(	call(A)
	*->	call(B)
	;	call(C)
	).


inv(false, true).
inv(true, false).


+(A, B, C) :-
	plus(A, B, C).


':-'(A, B) :-
	(	var(A)
	->	cpred(C),
		A =.. [C, _, _]
	;	true
	),
	clause(A, D),
	functor(A, P, 2),
	(	D =	(	(	compound(U)
				->	term_index(U, Ui),
					term_arg_1(U, Up)
				;	true
				),
				(	compound(V)
				->	term_index(V, Vi),
					term_arg_1(V, Vp)
				;	true
				),
				Y =.. [P, U, V, Ui, Vi, Up, Vp],
				call(Y)
			)
	->	call(D),
		B = true
	;	(	flag(nope)
		->	conj_cn(D, B)
		;	(	D = when(H, I)
			->	c_append(J, istep(Src, _, _, _), I),
				B = when(H, J)
			;	c_append(K, istep(Src, _, _, _), D),
				conj_cn(K, B)
			),
			term_index(':-'(A, B), Ind),
			(	\+prfstep(':-'(A, B), Ind, true, _, ':-'(A, B), _, forward, Src)
			->	assertz(prfstep(':-'(A, B), Ind, true, _, ':-'(A, B), _, forward, Src))
			;	true
			)
		)
	).


lookup(A, B, C) :-
	table(A, B, C),
	!.
lookup(A, B, C) :-
	var(A),
	nb_getval(table, M),
	N is M+1,
	nb_setval(table, N),
	atom_number(I, N),
	atomic_list_concat([B, '_table_entry_', I], A),
	assertz(table(A, B, C)).


escape_string([], []) :-
	!.
escape_string([0'\t|A], [0'\\, 0't|B]) :-
	!,
	escape_string(A, B).
escape_string([0'\b|A], [0'\\, 0'b|B]) :-
	!,
	escape_string(A, B).
escape_string([0'\n|A], [0'\\, 0'n|B]) :-
	!,
	escape_string(A, B).
escape_string([0'\r|A], [0'\\, 0'r|B]) :-
	!,
	escape_string(A, B).
escape_string([0'\f|A], [0'\\, 0'f|B]) :-
	!,
	escape_string(A, B).
escape_string([0'"|A], [0'\\, 0'"|B]) :-
	!,
	escape_string(A, B).
escape_string([0'\\|A], [0'\\, 0'\\|B]) :-
	!,
	escape_string(A, B).
escape_string([A|B], [A|C]) :-
	escape_string(B, C).


escape_squote([], []) :-
	!.
escape_squote([0''|A], [0'\\, 0''|B]) :-
	!,
	escape_squote(A, B).
escape_squote([A|B], [A|C]) :-
	escape_squote(B, C).


escape_unicode([], []) :-
	!.
escape_unicode([A, B|C], D) :-
	0xD800 =< A,
	A =< 0xDBFF,
	0xDC00 =< B,
	B =< 0xDFFF,
	E is 0x10000+(A-0xD800)*0x400+(B-0xDC00),
	(	0x100000 =< E
	->	with_output_to(codes(F), format('\\U00~16R', [E]))
	;	with_output_to(codes(F), format('\\U000~16R', [E]))
	),
	append(F, G, D),
	!,
	escape_unicode(C, G).
escape_unicode([A|B], [A|C]) :-
	escape_unicode(B, C).


quant(A, some) :-
	var(A),
	!.
quant('<http://www.w3.org/2000/10/swap/log#implies>'(_, _), allv) :-
	!.
quant(':-'(_, _), allv) :-
	!.
quant(answer('<http://www.w3.org/2000/10/swap/log#implies>', _, _, _, _, _, _), allv) :-
	!.
quant(answer(':-', _, _, _, _, _, _), allv) :-
	!.
quant(answer('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#tactic>', _, _, _, _, _, _), allv) :-
	!.
quant(_, some).


labelvars(A, B, C) :-
	quant(A, Q),
	labelvars(A, B, C, Q).


labelvars(A, B, C, D) :-
	var(A),
	!,
	atom_number(E, B),
	(	D == skolem
	->	nb_getval(var_ns, Vns),
		atomic_list_concat(['<', Vns, 'sk_', E, '>'], A)
	;	atomic_list_concat([D, E], A)
	),
	C is B+1.
labelvars(A, B, B, _) :-
	atomic(A),
	!.
labelvars(cn([A|B]), C, D, Q) :-
	!,
	labelvars(A, C, E, Q),
	(	B = [F]
	->	true
	;	F = cn(B)
	),
	labelvars(F, E, D, Q).
labelvars([A|B], C, D, Q) :-
	!,
	labelvars(A, C, E, Q),
	labelvars(B, E, D, Q).
labelvars(A, B, C, Q) :-
	nonvar(A),
	functor(A, _, D),
	labelvars(0, D, A, B, C, Q).


labelvars(A, A, _, B, B, _) :-
	!.
labelvars(A, B, C, D, E, Q) :-
	F is A+1,
	arg(F, C, G),
	labelvars(G, D, H, Q),
	labelvars(F, B, C, H, E, Q).


relabel([], []) :-
	!.
relabel([A|B], [C|D]) :-
	!,
	relabel(A, C),
	relabel(B, D).
relabel(A, B) :-
	atom(A),
	!,
	(	'<http://eulersharp.sourceforge.net/2003/03swap/log-rules#relabel>'(A, B)
	->	labelvars(B, 0, _)
	;	B = A
	).
relabel(A, A) :-
	number(A),
	!.
relabel(A, B) :-
	A =.. [C|D],
	relabel(C, E),
	relabel(D, F),
	B =.. [E|F].


partconc(_, [], []).
partconc(_, ['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(A, B)], ['<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(A, B)]) :-
	!.
partconc(A, [B|C], [B|D]) :-
	B = answer(E, _, _, _, _, _, _),
	(	E == '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#csvTuple>'
	;	E == '<http://www.w3.org/2000/10/swap/log#implies>'
	;	E == ':-'
	),
	!,
	partconc(A, C, D).
partconc(A, [B|C], [B|D]) :-
	commonvars(A, B, []),
	!,
	partconc(A, C, D).
partconc(A, [_|C], D) :-
	partconc(A, C, D).


commonvars('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#transaction>'(A, _), B, C) :-
	!,
	commonvars(A, B, C).
commonvars(A, B, C) :-
	term_variables(A, D),
	term_variables(B, E),
	copy_term_nat([D, E], [F, G]),
	labelvars([F, G], 0, _),
	findall(H,
		(	member(H, F),
			member(H, G)
		),
		C
	).


getvars(A, B) :-
	findvars(A, C, alpha),
	distinct(C, B).


makevars(A, B, Z) :-
	findvars(A, C, Z),
	distinct(C, D),
	length(D, E),
	length(F, E),
	makevars(A, B, D, F).


makevars(A, B, C, D) :-
	atomic(A),
	!,
	(	atom(A),
		nth0(E, C, A)
	->	nth0(E, D, B)
	;	B = A
	).
makevars(A, A, _, _) :-
	var(A),
	!.
makevars([], [], _, _) :-
	!.
makevars([A|B], [C|D], E, F) :-
	makevars(A, C, E, F),
	makevars(B, D, E, F),
	!.
makevars(A, B, E, F) :-
	A =.. C,
	makevars(C, D, E, F),
	B =.. D.


findvars(A, B, Z) :-
	atomic(A),
	!,
	(	atom(A),
		findvar(A, Z)
	->	B = [A]
	;	B = []
	).
findvars(A, [], _) :-
	var(A),
	!.
findvars([], [], _) :-
	!.
findvars([A|B], C, Z) :-
	findvars(A, D, Z),
	findvars(B, E, Z),
	append(D, E, C),
	!.
findvars(A, B, Z) :-
	A =.. C,
	findvars(C, B, Z).


findvar(A, alpha) :-
	!,
	nb_getval(var_ns, Vns),
	sub_atom(A, 1, _, _, Vns).
findvar(A, beta) :-
	!,
	(	nb_getval(var_ns, Vns),
		sub_atom(A, 1, _, _, Vns)
	;	atom_concat('_bn_', _, A)
	;	atom_concat('_e_', _, A)
	;	atom_concat(some, _, A)
	).
findvar(A, gamma) :-
	!,
	sub_atom(A, _, 19, _, '/.well-known/genid/'),
	\+sub_atom(A, _, 4, _, '#bn_'),
	\+sub_atom(A, _, 3, _, '#e_').
findvar(A, delta) :-
	!,
	(	sub_atom(A, _, 19, _, '/.well-known/genid/')
	;	atom_concat(some, _, A)
	).
findvar(A, epsilon) :-
	sub_atom(A, 0, 1, _, '_'),
	\+atom_concat('_bn_', _, A),
	\+atom_concat('_e_', _, A).


raw_type(A, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#List>') :-
	is_list(A),
	!.
raw_type(A, '<http://www.w3.org/2000/01/rdf-schema#Literal>') :-
	number(A),
	!.
raw_type(A, '<http://www.w3.org/2000/01/rdf-schema#Literal>') :-
	atom(A),
	\+ atom_concat(some, _, A),
	\+ (sub_atom(A, 0, 1, _, '<'), sub_atom(A, _, 1, 0, '>')),
	!.
raw_type(literal(_, _), '<http://www.w3.org/2000/01/rdf-schema#Literal>') :-
	!.
raw_type(rdiv(_, _), '<http://www.w3.org/2000/01/rdf-schema#Literal>') :-
	!.
raw_type('<http://eulersharp.sourceforge.net/2003/03swap/log-rules#epsilon>', '<http://www.w3.org/2000/01/rdf-schema#Literal>') :-
	!.
raw_type(cn(_), '<http://www.w3.org/2000/10/swap/log#Formula>') :-
	!.
raw_type(set(_), '<http://www.w3.org/2000/10/swap/log#Set>') :-
	!.
raw_type(A, '<http://www.w3.org/2000/10/swap/log#Formula>') :-
	functor(A, B, C),
	B \= ':',
	C >= 2,
	!.
raw_type(_, '<http://www.w3.org/2000/10/swap/log#Other>').


getnumber(rdiv(A, B), C) :-
	nonvar(A),
	!,
	C is A/B.
getnumber(A, A) :-
	number(A),
	!.
getnumber(A, epsilon) :-
	nonvar(A),
	A = '<http://eulersharp.sourceforge.net/2003/03swap/log-rules#epsilon>',
	!.
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#dateTime>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	datetime(B, C, []).
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#date>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	date(B, C, []).
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#time>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	time(B, C, []).
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#duration>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	duration(B, C, []).
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#yearMonthDuration>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	yearmonthduration(B, C, []).
getnumber(literal(A, type('<http://www.w3.org/2001/XMLSchema#dayTimeDuration>')), B) :-
	!,
	ground(A),
	atom_codes(A, C),
	daytimeduration(B, C, []).
getnumber(literal(A, _), B) :-
	ground(A),
	atom_codes(A, C),
	numeral(C, D),
	catch(number_codes(B, D), _, fail).


getint(A, B) :-
	getnumber(A, C),
	B is integer(round(C)).


getbool(literal(false, type('<http://www.w3.org/2001/XMLSchema#boolean>')), false).
getbool(literal(true, type('<http://www.w3.org/2001/XMLSchema#boolean>')), true).
getbool(literal('0', type('<http://www.w3.org/2001/XMLSchema#boolean>')), false).
getbool(literal('1', type('<http://www.w3.org/2001/XMLSchema#boolean>')), true).
getbool(false, false).
getbool(true, true).


getlist(A, A) :-
	var(A),
	!.
getlist(set(A), A) :-
	!.
getlist('<http://www.w3.org/1999/02/22-rdf-syntax-ns#nil>', []) :-
	!.
getlist([], []) :-
	!.
getlist([A|B], [C|D]) :-
	getlist(A, C),
	!,
	getlist(B, D).
getlist([A|B], [A|D]) :-
	!,
	getlist(B, D).
getlist(A, [B|C]) :-
	'<http://www.w3.org/1999/02/22-rdf-syntax-ns#first>'(A, B),
	'<http://www.w3.org/1999/02/22-rdf-syntax-ns#rest>'(A, D),
	getlist(D, C).


getstring(A, B) :-
	'<http://www.w3.org/2000/10/swap/log#uri>'(A, B),
	!.
getstring(A, A).


getcodes(literal(A, _), B) :-
	nonvar(A),
	!,
	atom_codes(A, B).
getcodes(A, B) :-
	with_output_to_chars(wg(A), B).


preformat([], []) :-
	!.
preformat([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>'))|B], [A|D]) :-
	!,
	preformat(B, D).
preformat([A|B], [A|D]) :-
	preformat(B, D).


numeral([0'-, 0'.|A], [0'-, 0'0, 0'.|A]) :-
	!.
numeral([0'+, 0'.|A], [0'+, 0'0, 0'.|A]) :-
	!.
numeral([0'.|A], [0'0, 0'.|A]) :-
	!.
numeral(A, B) :-
	append([C, [0'., 0'e], D], A),
	append([C, [0'., 0'0, 0'e], D], B),
	!.
numeral(A, B) :-
	append([C, [0'., 0'E], D], A),
	append([C, [0'., 0'0, 0'E], D], B),
	!.
numeral(A, B) :-
	last(A, 0'.),
	append(A, [0'0], B),
	!.
numeral(A, A).


rdiv_codes(rdiv(A, B), C) :-
	append(D, [0'.|E], C),
	append(D, E, F),
	number_codes(A, F),
	lzero(E, G),
	number_codes(B, [0'1|G]),
	!.
rdiv_codes(rdiv(A, 1), C) :-
	number_codes(A, C).


lzero([], []) :-
	!.
lzero([_|A], [0'0|B]) :-
	lzero(A, B).


dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#integer>'], B) :-
	integer(B),
	!,
	atom_number(A, B).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#double>'], B) :-
	float(B),
	!,
	atom_number(A, B).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#dateTime>'], B) :-
	(	number(B)
	->	datetime(B, C)
	;	nonvar(B),
		B = date(Year, Month, Day, Hour, Minute, Second, Offset, _, _),
		datetime(Year, Month, Day, Hour, Minute, Second, Offset, C)
	),
	!,
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#date>'], B) :-
	(	number(B)
	->	date(B, C)
	;	nonvar(B),
		B = date(Year, Month, Day, _, _, _, Offset, _, _),
		date(Year, Month, Day, Offset, C)
	),
	!,
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#time>'], B) :-
	(	number(B)
	->	time(B, C)
	;	nonvar(B),
		B = date(_, _, _, Hour, Minute, Second, Offset, _, _),
		time(Hour, Minute, Second, Offset, C)
	),
	!,
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#duration>'], B) :-
	number(B),
	!,
	daytimeduration(B, C),
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#yearMonthDuration>'], B) :-
	number(B),
	!,
	yearmonthduration(B, C),
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#dayTimeDuration>'], B) :-
	number(B),
	!,
	daytimeduration(B, C),
	atom_codes(A, C).
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#boolean>'], A) :-
	atomic(A),
	getbool(A, A),
	!.
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), prolog:atom], A) :-
	atomic(A),
	\+sub_atom(A, 0, 1, _, '<'),
	!.
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), '<http://www.w3.org/2001/XMLSchema#string>'], literal(A, lang(_))) :-
	!.
dtlit([literal(A, type('<http://www.w3.org/2001/XMLSchema#string>')), B], literal(A, type(B))).


hash_to_ascii([], L1, L1).
hash_to_ascii([A|B], [C, D|L3], L4) :-
	E is A>>4 /\ 15,
	F is A /\ 15,
	code_type(C, xdigit(E)),
	code_type(D, xdigit(F)),
	hash_to_ascii(B, L3, L4).


:- if(\+current_predicate(get_time/1)).
get_time(A) :-
	datime(B),
	mktime(B, C),
	A is C*1.0.
:- endif.


memotime(datime(A, B, C, D, E, F), G) :-
	(	mtime(datime(A, B, C, D, E, F), G)
	->	true
	;	date_time_stamp(date(A, B, C, D, E, F, 0, -, -), H),
		fmsec(F, H, G),
		assertz(mtime(datime(A, B, C, D, E, F), G))
	).


datetime(A, L1, L13) :-
	int(B, L1, [0'-|L3]),
	int(C, L3, [0'-|L5]),
	int(D, L5, [0'T|L7]),
	int(E, L7, [0':|L9]),
	int(F, L9, [0':|L11]),
	decimal(G, L11, L12),
	timezone(H, L12, L13),
	I is -H,
	date_time_stamp(date(B, C, D, E, F, G, I, -, -), J),
	fmsec(G, J, A).


datetime(A, B, C, D, E, F, G, L1, L13) :-
	int(A, L1, [0'-|L3]),
	int(B, L3, [0'-|L5]),
	int(C, L5, [0'T|L7]),
	int(D, L7, [0':|L9]),
	int(E, L9, [0':|L11]),
	decimal(F, L11, L12),
	timezone(G, L12, L13).


date(A, L1, L7) :-
	int(B, L1, [0'-|L3]),
	int(C, L3, [0'-|L5]),
	int(D, L5, L6),
	timezone(H, L6, L7),
	I is -H,
	date_time_stamp(date(B, C, D, 0, 0, 0, I, -, -), E),
	fmsec(0, E, A).


date(A, B, C, D, L1, L7) :-
	int(A, L1, [0'-|L3]),
	int(B, L3, [0'-|L5]),
	int(C, L5, L6),
	timezone(D, L6, L7).


time(A, L1, L7) :-
	int(B, L1, [0':|L3]),
	int(C, L3, [0':|L5]),
	decimal(D, L5, L6),
	timezone(E, L6, L7),
	(	B = 24
	->	A is C*60+D-E
	;	A is B*3600+C*60+D-E
	).


time(A, B, C, D, L1, L7) :-
	int(A, L1, [0':|L3]),
	int(B, L3, [0':|L5]),
	decimal(C, L5, L6),
	timezone(D, L6, L7).


duration(A, L1, L7) :-
	dsign(B, L1, [0'P|L3]),
	years(C, L3, L4),
	months(D, L4, L5),
	days(E, L5, L6),
	dtime(F, L6, L7),
	A is B*(C*31556952+D*2629746+E*86400.0+F).


yearmonthduration(A, L1, L5) :-
	dsign(B, L1, [0'P|L3]),
	years(C, L3, L4),
	months(D, L4, L5),
	A is B*(C*12+D).


daytimeduration(A, L1, L5) :-
	dsign(B, L1, [0'P|L3]),
	days(C, L3, L4),
	dtime(D, L4, L5),
	A is B*(C*86400.0+D).


timezone(A, L1, L4) :-
	int(B, L1, [0':|L3]),
	!,
	int(C, L3, L4),
	A is B*3600+C*60.
timezone(0, [0'Z|L2], L2) :-
	!.
timezone(0, L1, L1).


dsign(1, [0'+|L2], L2).
dsign(-1, [0'-|L2], L2).
dsign(1, L1, L1).


dtime(A, [0'T|L2], L5) :-
	!,
	hours(B, L2, L3),
	minutes(C, L3, L4),
	seconds(D, L4, L5),
	A is B*3600+C*60+D.
dtime(0, L1, L1).


years(A, L1, L3) :-
	int(A, L1, [0'Y|L3]).
years(0, L1, L1).


months(A, L1, L3) :-
	int(A, L1, [0'M|L3]).
months(0, L1, L1) .


days(A, L1, L3) :-
	int(A, L1, [0'D|L3]).
days(0, L1, L1).


hours(A, L1, L3) :-
	int(A, L1, [0'H|L3]).
hours(0, L1, L1).


minutes(A, L1, L3) :-
	int(A, L1, [0'M|L3]).
minutes(0, L1, L1).


seconds(A, L1, L3) :-
	decimal(A, L1, [0'S|L3]).
seconds(0, L1, L1).


int(A, L1, L4) :-
	sgn(B, L1, L2),
	digit(C, L2, L3),
	digits(D, L3, L4),
	number_codes(A, [B, C|D]).


decimal(A, L1, L5) :-
	sgn(B, L1, L2),
	digit(C, L2, L3),
	digits(D, L3, L4),
	fraction(E, L4, L5),
	append([B, C|D], E, F),
	number_codes(A, F).


sgn(0'+, [0'+|L2], L2).
sgn(0'-, [0'-|L2], L2).
sgn(0'+, L1, L1).


fraction([0'., A|B], [0'.|L2], L4) :-
	!,
	digit(A, L2, L3),
	digits(B, L3, L4).
fraction([], L1, L1).


digits([A|B], L1, L3) :-
	digit(A, L1, L2),
	digits(B, L2, L3).
digits([], L1, L1).


digit(A, [A|L2], L2) :-
	code_type(A, digit).


fmsec(A, B, C) :-
	integer(A),
	!,
	C is floor(B).
fmsec(_, B, B).


datetime(A, B) :-
	stamp_date_time(A, date(Year, Month, Day, Hour, Minute, Second, _, _, _), 0),
	fmsec(A, Second, Sec),
	ycodes(Year, C),
	ncodes(Month, D),
	ncodes(Day, E),
	ncodes(Hour, F),
	ncodes(Minute, G),
	ncodes(Sec, H),
	append([C, [0'-], D, [0'-], E, [0'T], F, [0':], G, [0':], H, [0'Z]], B).


datetime(Year, Month, Day, Hour, Minute, Second, Offset, B) :-
	ycodes(Year, C),
	ncodes(Month, D),
	ncodes(Day, E),
	ncodes(Hour, F),
	ncodes(Minute, G),
	ncodes(Second, H),
	(	Offset =:= 0
	->	append([C, [0'-], D, [0'-], E, [0'T], F, [0':], G, [0':], H, [0'Z]], B)
	;	(	Offset > 0
		->	I = [0'-],
			OHour is Offset//3600
		;	I = [0'+],
			OHour is -Offset//3600
		),
		ncodes(OHour, J),
		OMinute is (Offset mod 3600)//60,
		ncodes(OMinute, K),
		append([C, [0'-], D, [0'-], E, [0'T], F, [0':], G, [0':], H, I, J, [0':], K], B)
	).


date(A, B) :-
	N is A+3600*12,
	stamp_date_time(N, date(Year, Month, Day, _, _, _, _, _, _), 0),
	ycodes(Year, C),
	ncodes(Month, D),
	ncodes(Day, E),
	Offset is (round(floor(N)) mod 86400) - 3600*12,
	(	Offset =:= 0
	->	append([C, [0'-], D, [0'-], E, [0'Z]], B)
	;	(	Offset > 0
		->	I = [0'-],
			OHour is Offset//3600
		;	I = [0'+],
			OHour is -Offset//3600
		),
		ncodes(OHour, J),
		OMinute is (Offset mod 3600)//60,
		ncodes(OMinute, K),
		append([C, [0'-], D, [0'-], E, I, J, [0':], K], B)
	).


date(Year, Month, Day, Offset, B) :-
	ycodes(Year, C),
	ncodes(Month, D),
	ncodes(Day, E),
	(	Offset =:= 0
	->	append([C, [0'-], D, [0'-], E, [0'Z]], B)
	;	(	Offset > 0
		->	I = [0'-],
			OHour is Offset//3600
		;	I = [0'+],
			OHour is -Offset//3600
		),
		ncodes(OHour, J),
		OMinute is (Offset mod 3600)//60,
		ncodes(OMinute, K),
		append([C, [0'-], D, [0'-], E, I, J, [0':], K], B)
	).


time(A, B) :-
	stamp_date_time(A, date(_, _, _, Hour, Minute, Second, _, _, _), 0),
	fmsec(A, Second, Sec),
	ncodes(Hour, F),
	ncodes(Minute, G),
	ncodes(Sec, H),
	append([F, [0':], G, [0':], H, [0'Z]], B).


time(Hour, Minute, Second, Offset, B) :-
	ncodes(Hour, F),
	ncodes(Minute, G),
	ncodes(Second, H),
	(	Offset =:= 0
	->	append([F, [0':], G, [0':], H, [0'Z]], B)
	;	(	Offset > 0
		->	I = [0'-],
			OHour is Offset//3600
		;	I = [0'+],
			OHour is -Offset//3600
		),
		ncodes(OHour, J),
		OMinute is (Offset mod 3600)//60,
		ncodes(OMinute, K),
		append([F, [0':], G, [0':], H, I, J, [0':], K], B)
	).


yearmonthduration(A, B) :-
	(	A < 0
	->	C = [0'-]
	;	C = []
	),
	D is abs(A),
	E is D//12,
	number_codes(E, Years),
	F is D-(D//12)*12,
	number_codes(F, Months),
	append([C, [0'P], Years, [0'Y], Months, [0'M]], B).


daytimeduration(A, B) :-
	AInt is round(floor(A)),
	AFrac is A-AInt,
	(	AInt < 0
	->	C = [0'-]
	;	C = []
	),
	D is abs(AInt),
	E is D//86400,
	number_codes(E, Days),
	F is (D-(D//86400)*86400)//3600,
	number_codes(F, Hours),
	G is (D-(D//3600)*3600)//60,
	number_codes(G, Minutes),
	H is D-(D//60)*60+AFrac,
	number_codes(H, Seconds),
	append([C, [0'P], Days, [0'D, 0'T], Hours, [0'H], Minutes, [0'M], Seconds, [0'S]], B).


ncodes(A, B) :-
	number_codes(A, D),
	(	A < 10
	->	append([[0'0], D], B)
	;	B = D
	).


ycodes(A, B) :-
	C is abs(A),
	number_codes(C, D),
	(	C < 10
	->	append([[0'0, 0'0, 0'0], D], E)
	;	(	C < 100
		->	append([[0'0, 0'0], D], E)
		;	(	C < 1000
			->	append([[0'0], D], E)
			;	E = D
			)
		)
	),
	(	A >= 0
	->	B = E
	;	B = [0'-|E]
	).


absolute_uri('-', '-') :-
	!.
absolute_uri(A, B) :-
	(	is_absolute_url(A)
	->	B = A
	;	absolute_file_name(A, C),
		prolog_to_os_filename(D, C),
		atom_codes(D, E),
		subst([[[0x20], [0'%, 0'2, 0'0]]], E, F),
		atom_codes(G, F),
		(	current_prolog_flag(windows, true)
		->	atomic_list_concat(['file:///', G], B)
		;	atomic_list_concat(['file://', G], B)
		)
	).


:- if(current_predicate(uri_resolve/3)).
resolve_uri(A, _, A) :-
	uri_is_global(A),
	!.
resolve_uri('', A, A) :-
	!.
resolve_uri(A, B, C) :-
	sub_atom(A, 0, 1, _, '?'),
	(	sub_atom(B, I, 1, _, '?')
	->	true
	;	atom_length(B, I)
	),
	sub_atom(B, 0, I, _, D),
	atomic_list_concat([D, A], C),
	!.
resolve_uri(A, B, C) :-
	sub_atom(A, 0, 1, _, '#'),
	(	sub_atom(B, I, 1, _, '#')
	->	true
	;	atom_length(B, I)
	),
	sub_atom(B, 0, I, _, D),
	atomic_list_concat([D, A], C),
	!.
resolve_uri(A, B, C) :-
	uri_resolve(A, B, C).
:-else.
resolve_uri(A, _, A) :-
	sub_atom(A, _, 1, _, ':'),
	!.
resolve_uri('', A, A) :-
	!.
resolve_uri('#', A, B) :-
	!,
	atomic_list_concat([A, '#'], B).
resolve_uri(A, B, A) :-
	\+sub_atom(B, _, 1, _, ':'),
	!.
resolve_uri(A, B, C) :-
	so_uri(U),
	atom_length(U, V),
	sub_atom(A, 0, 1, _, '#'),
	sub_atom(B, 0, V, _, U),
	!,
	atomic_list_concat([B, A], C).
resolve_uri(A, B, C) :-
	sub_atom(A, 0, 2, _, './'),
	!,
	sub_atom(A, 2, _, 0, R),
	resolve_uri(R, B, C).
resolve_uri(A, B, C) :-
	sub_atom(A, 0, 3, _, '../'),
	!,
	sub_atom(A, 3, _, 0, R),
	so_uri(U),
	atom_length(U, V),
	sub_atom(B, 0, V, D, U),
	sub_atom(B, V, D, _, E),
	(	sub_atom(E, F, 1, G, '/'),
		sub_atom(E, _, G, 0, H),
		\+sub_atom(H, _, _, _, '/'),
		K is V+F
	->	sub_atom(B, 0, K, _, S)
	;	S = B
	),
	resolve_uri(R, S, C).
resolve_uri(A, B, C) :-
	so_uri(U),
	atom_length(U, V),
	sub_atom(A, 0, 1, _, '/'),
	sub_atom(B, 0, V, D, U),
	sub_atom(B, V, D, _, E),
	(	sub_atom(E, F, 1, _, '/')
	->	sub_atom(E, 0, F, _, G)
	;	G = E
	),
	!,
	atomic_list_concat([U, G, A], C).
resolve_uri(A, B, C) :-
	so_uri(U),
	atom_length(U, V),
	sub_atom(B, 0, V, D, U),
	sub_atom(B, V, D, _, E),
	(	sub_atom(E, F, 1, G, '/'),
		sub_atom(E, _, G, 0, H),
		\+sub_atom(H, _, _, _, '/')
	->	sub_atom(E, 0, F, _, I)
	;	I = E
	),
	!,
	atomic_list_concat([U, I, '/', A], C).
resolve_uri(A, _, _) :-
	nb_getval(line_number, Ln),
	throw(unresolvable_relative_uri(A, after_line(Ln))).
:- endif.


so_uri('http://').
so_uri('https://').
so_uri('ftp://').
so_uri('file://').


wcacher(A, B) :-
	wcache(A, B),
	!.
wcacher(A, B) :-
	wcache(C, D),
	sub_atom(A, 0, I, _, C),
	sub_atom(A, I, _, 0, E),
	atomic_list_concat([D, E], B).


prolog_verb(S, Name) :-
	(	atom(S),
		atom_concat('\'<http://eulersharp.sourceforge.net/2003/03swap/prolog#', A, S),
		atom_concat(B, '>\'', A)
	->	(	B = conjunction
		->	Pred = '\',\''
		;	(	B = disjunction
			->	Pred = '\';\''
			;	(	prolog_sym(B, Pred, _)
				->	true
				;	nb_getval(line_number, Ln),
					throw(invalid_prolog_builtin(B, after_line(Ln)))
				)
			)
		),
		Name = prolog:Pred
	;	Name = S
	).


dynamic_verb(Verb) :-
	(	(	atom(Verb)
		->	V = Verb
		;	Verb = isof(V)
		),
		\+sub_atom(V, 0, 1, _, '_')
	->	(	intern(V)
		->	true
		;	assertz(intern(V)),
			(	sub_atom(V, 0, 1, _, '\'')
			->	sub_atom(V, 1, _, 1, A),
				(	sub_atom(A, _, 1, _, '\'')
				->	atom_codes(A, C),
					escape_squote(D, C),
					atom_codes(B, D)
				;	B = A
				)
			;	B = V
			),
			(	current_predicate(B/2)
			->	true
			;	dynamic(B/2),
				(	flag(n3p)
				->	format('dynapred(~q).~n', [B/2])
				;	true
				)
			)
		)
	;	true
	).


timestamp(Stamp) :-
	get_time(StampN),
	datetime(StampN, StampC),
	atom_codes(StampA, StampC),
	(	sub_atom(StampA, I, 1, 0, 'Z'),
		I > 23
	->	sub_atom(StampA, 0, 23, _, StampB),
		atomic_list_concat([StampB, 'Z'], Stamp)
	;	Stamp = StampA
	).


fm(A) :-
	format(user_error, '*** ~q~n', [A]),
	flush_output(user_error).


mf(A) :-
	forall(
		(	call(A)
		),
		(	format(user_error, '*** ~q~n', [A])
		)
	),
	flush_output(user_error).


% Regular Expressions inspired by http://www.cs.sfu.ca/~cameron/Teaching/384/99-3/regexp-plg.html

regex(RE_esc, Input_esc, Outputs_esc) :-
	escape_string(RE, RE_esc),
	re(Parsed_RE, RE, []),
	(	RE = [0'^|_]
	->	Bos = true
	;	Bos = false
	),
	escape_string(Input, Input_esc),
	tokenize2(Parsed_RE, Input, Outputs, Bos),
	findall(Output_esc,
		(	member(Output, Outputs),
			escape_string(Output, Output_esc)
		),
		Outputs_esc
	),
	!.


tokenize2(_P_RE, [], [], true).
tokenize2(P_RE, Input, Output, Bos) :-
	(	rematch1(P_RE, Input, _, Output)
	->	true
	;	Bos = false,
		Input = [_|Inp],
		tokenize2(P_RE, Inp, Output, Bos)
	).


rematch1(union(RE1, _RE2), S, U, Selected) :-
	rematch1(RE1, S, U, Selected).
rematch1(union(_RE1, RE2), S, U, Selected) :-
	rematch1(RE2, S, U, Selected).
rematch1(conc(RE1, RE2), S, U, Selected) :-
	rematch1(RE1, S, U1, Sel1),
	rematch1(RE2, U1, U, Sel2),
	append(Sel1, Sel2, Selected).
rematch1(star(RE), S, U, Selected) :-
	rematch1(RE, S, U1, Sel1),
	rematch1(star(RE), U1, U, Sel2),
	append(Sel1, Sel2, Selected).
rematch1(star(_RE), S, S, []).
rematch1(qm(RE), S, U, Selected) :-
	rematch1(RE, S, U, Selected).
rematch1(qm(_RE), S, S, []).
rematch1(plus(RE), S, U, Selected) :-
	rematch1(RE, S, U1, Sel1),
	rematch1(star(RE), U1, U, Sel2),
	append(Sel1, Sel2, Selected).
rematch1(group(RE), S, U, Selected) :-
	rematch1(RE, S, U, Sel1),
	append(P, U, S),
	append(Sel1, [P], Selected).
rematch1(any, [_C1|U], U, []).
rematch1(char(C), [C|U], U, []).
rematch1(bos, S, S, []).
rematch1(eos, [], [], []).
rematch1(negSet(Set), [C|U], U, []) :-
	\+charSetMember(C, Set).
rematch1(posSet(Set), [C|U], U, []) :-
	charSetMember(C, Set).


charSetMember(C, [char(C)|_]).
charSetMember(C, [range(C1, C2)|_]) :-
	C1 =< C,
	C =< C2.
charSetMember(C, [negSet(Set)|_]) :-
	\+charSetMember(C, Set).
charSetMember(C, [posSet(Set)|_]) :-
	charSetMember(C, Set).
charSetMember(C, [_|T]) :-
	charSetMember(C, T).


re(Z, L1, L3) :-
	basicRE(W, L1, L2),
	reTail(W, Z, L2, L3).


reTail(W, Z, [0'||L2], L4) :-
	basicRE(X, L2, L3),
	reTail(union(W, X), Z, L3, L4).
reTail(W, W, L1, L1).


basicRE(Z, L1, L3) :-
	simpleRE(W, L1, L2),
	basicREtail(W, Z, L2, L3).


basicREtail(W, Z, L1, L3) :-
	simpleRE(X, L1, L2),
	basicREtail(conc(W, X), Z, L2, L3).
basicREtail(W, W, L1, L1).


simpleRE(Z, L1, L3) :-
	elementalRE(W, L1, L2),
	simpleREtail(W, Z, L2, L3).


simpleREtail(W, star(W), [0'*|L2], L2).
simpleREtail(W, qm(W), [0'?|L2], L2).
simpleREtail(W, plus(W), [0'+|L2], L2).
simpleREtail(W, W, L1, L1).


elementalRE(any, [0'.|L2], L2).
elementalRE(group(X), [0'(|L2], L4) :-
	re(X, L2, [0')|L4]).
elementalRE(bos, [0'^|L2], L2).
elementalRE(eos, [0'$|L2], L2).
elementalRE(posSet([range(0'A, 0'Z), range(0'a, 0'z), range(0'0, 0'9), char(0'_)]), [0'\\, 0'w|L2], L2).
elementalRE(negSet([range(0'A, 0'Z), range(0'a, 0'z), range(0'0, 0'9), char(0'_)]), [0'\\, 0'W|L2], L2).
elementalRE(posSet([range(0'0, 0'9)]), [0'\\, 0'd|L2], L2).
elementalRE(negSet([range(0'0, 0'9)]), [0'\\, 0'D|L2], L2).
elementalRE(posSet([char(0x20), char(0'\t), char(0'\r), char(0'\n), char(0'\v), char(0'\f)]), [0'\\, 0's|L2], L2).
elementalRE(negSet([char(0x20), char(0'\t), char(0'\r), char(0'\n), char(0'\v), char(0'\f)]), [0'\\, 0'S|L2], L2).
elementalRE(char(C), [0'\\, C|L2], L2) :-
	re_metachar([C]).
elementalRE(char(C), [C|L2], L2) :-
	\+re_metachar([C]).
elementalRE(negSet(X), [0'[, 0'^|L2], L4) :-
	!,
	setItems(X, L2, [0']|L4]).
elementalRE(posSet(X), [0'[|L2], L4) :-
	setItems(X, L2, [0']|L4]).


re_metachar([0'\\]).
re_metachar([0'|]).
re_metachar([0'*]).
re_metachar([0'?]).
re_metachar([0'+]).
re_metachar([0'.]).
re_metachar([0'[]).
re_metachar([0'$]).
re_metachar([0'(]).
re_metachar([0')]).


setItems([Item1|MoreItems], L1, L3) :-
	setItem(Item1, L1, L2),
	setItems(MoreItems, L2, L3).
setItems([Item1], L1, L2) :-
	setItem(Item1, L1, L2).


setItem(posSet([range(0'A, 0'Z), range(0'a, 0'z), range(0'0, 0'9), char(0'_)]), [0'\\, 0'w|L2], L2).
setItem(negSet([range(0'A, 0'Z), range(0'a, 0'z), range(0'0, 0'9), char(0'_)]), [0'\\, 0'W|L2], L2).
setItem(posSet([range(0'0, 0'9)]), [0'\\, 0'd|L2], L2).
setItem(negSet([range(0'0, 0'9)]), [0'\\, 0'D|L2], L2).
setItem(posSet([char(0x20), char(0'\t), char(0'\r), char(0'\n), char(0'\v), char(0'\f)]), [0'\\, 0's|L2], L2).
setItem(negSet([char(0x20), char(0'\t), char(0'\r), char(0'\n), char(0'\v), char(0'\f)]), [0'\\, 0'S|L2], L2).
setItem(char(C), [0'\\, C|L2], L2) :-
	set_metachar([C]).
setItem(char(C), [C|L2], L2) :-
	\+set_metachar([C]).
setItem(range(A, B), L1, L4) :-
	setItem(char(A), L1, [0'-|L3]),
	setItem(char(B), L3, L4).


set_metachar([0'\\]).
set_metachar([0']]).
set_metachar([0'-]).


regexp_wildcard([], []) :-
	!.
regexp_wildcard([0'*|A], [0'., 0'*|B]) :-
	!,
	regexp_wildcard(A, B).
regexp_wildcard([A|B], [A|C]) :-
	regexp_wildcard(B, C).

