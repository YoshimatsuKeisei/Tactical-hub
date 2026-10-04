const unitRows = [
  ["王", "3", "1", "1", "撃破されるとチーム敗北"],
  ["歩兵", "1", "1", "1", "基本兵種"],
  ["重歩兵", "2", "1", "1", "歩兵2体の合流で編成"],
  ["騎兵", "1", "2", "1", "高い移動力"],
  ["弓兵", "1", "1", "3", "遠距離攻撃"],
  ["忍者", "1", "1", "1", "湖上移動・隠密"],
  ["工", "1", "1", "5", "拠点内の敵だけを攻撃"],
  ["軍師", "1", "1", "0", "攻撃せず特殊能力を使用"],
] as const;

const attackRows = [
  ["王", "1/6", "1/5", "1/5", "1/5", "1/5", "1/5", "1/5", "1/5"],
  ["歩兵", "1/7", "1/6", "1/6", "1/7", "1/5", "1/5", "1/5", "1/5"],
  ["重歩兵", "1/6", "1/5", "1/6", "1/6", "1/4", "1/4", "1/4", "1/4"],
  ["騎兵", "1/7", "1/5", "1/5", "1/6", "1/7", "1/5", "1/5", "1/5"],
  ["弓兵", "1/7", "1/7", "1/7", "1/5", "1/6", "1/5", "1/5", "1/5"],
  ["忍者", "1/7", "1/7", "1/7", "1/7", "1/7", "1/6", "1/5", "1/5"],
] as const;

function RuleSection({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return <article className="rules-section">
    <h3>{number}. {title}</h3>
    {children}
  </article>;
}

export function RulesScreen() {
  return <div className="rules-content">
    <p className="rules-introduction">
      初めて遊ぶ方向けに、ゲームの基本的な流れと確認済みの詳細ルールをまとめています。
    </p>

    <section className="rules-group" aria-labelledby="basic-rules-title">
      <h2 id="basic-rules-title">基本ルール</h2>

      <RuleSection number={1} title="ゲームの目的">
        <p>Tactical-hubは、道路・拠点・湖を巡って戦う4チーム対戦ゲームです。</p>
        <p>敵の王を倒し、敵の拠点を攻略しながら、最後まで活動可能なチームとして残ることを目指します。</p>
        <p>中立守備隊はプレイヤーとは別の勢力です。中立守備隊も戦闘に参加し、攻撃可能な敵を自動で攻撃します。</p>
      </RuleSection>

      <RuleSection number={2} title="ゲーム開始">
        <p>標準ゲームには4つのプレイヤーチームと中立守備隊が登場します。</p>
        <div className="rules-columns">
          <div>
            <h4>各プレイヤーの本拠地</h4>
            <ul>
              <li>王 1体</li>
              <li>鼓舞軍師 1体</li>
            </ul>
          </div>
          <div>
            <h4>5つの中立拠点</h4>
            <ul>
              <li>歩兵 1体</li>
              <li>騎兵 1体</li>
              <li>弓兵 1体</li>
            </ul>
          </div>
        </div>
        <p>各拠点には4つの駐留枠があります。</p>
      </RuleSection>

      <RuleSection number={3} title="ターンとフェーズ">
        <ol>
          <li>各チームの生産と移動</li>
          <li>攻撃先の選択</li>
          <li>全攻撃の同時解決</li>
          <li>発生した褒賞の配置</li>
          <li>必要な軍師能力の選択と同時解決</li>
          <li>次のターンへ</li>
        </ol>
        <p>移動を行うチームの順番はターンごとにローテーションします。敗北したチームは以後の行動順から除外されます。</p>
      </RuleSection>

      <RuleSection number={4} title="マップと道路">
        <p>通常の駒は、道路・拠点・使用可能な橋を使って移動・攻撃します。</p>
        <ul>
          <li>湖は通常の駒では移動できません。忍者は湖上を移動できます。</li>
          <li>橋は湖を越える道路として機能し、敵味方を問わず利用できます。</li>
          <li>障害物が置かれた場所は移動できません。</li>
          <li>障害物そのものに、攻撃を直接遮断する効果はありません。</li>
          <li>近接駒は障害物により接近できない結果として、攻撃できない場合があります。</li>
          <li>遠距離攻撃が、障害物だけを理由に遮断されることはありません。</li>
        </ul>
      </RuleSection>

      <RuleSection number={5} title="拠点">
        <p>各拠点には4つの駐留枠があります。</p>
        <p>本拠地には「奥座敷」があります。奥座敷にいる駒は、同じ本拠地の別枠に生存中の味方駒がいる間、攻撃対象にできません。</p>
      </RuleSection>

      <RuleSection number={6} title="移動">
        <p><strong>移動は即時に確定します。</strong></p>
        <ol>
          <li>駒を選択</li>
          <li>合法な移動先を選択</li>
          <li>その場で盤面へ反映</li>
        </ol>
        <p>各駒は1回の移動フェーズにつき1回だけ移動できます。先に行った移動は直ちに盤面へ反映され、更新後の盤面を使って次の駒の移動候補が決まります。</p>
        <p>通常移動には、将来の移動先を予約する仕組みはありません。</p>
      </RuleSection>

      <RuleSection number={7} title="攻撃と戦闘">
        <p>各チームの攻撃選択完了後、攻撃は完全同時に解決されます。</p>
        <ul>
          <li>命中した攻撃は通常1ダメージです。</li>
          <li>同じ駒が複数の攻撃を受けた場合、ダメージは合計されます。</li>
          <li>戦闘解決前に攻撃内容が確定するため、相討ちも発生します。</li>
          <li>中立守備隊は攻撃可能な敵を自動で選択します。</li>
        </ul>
      </RuleSection>

      <RuleSection number={8} title="生産">
        <p>生産ターンは1、6、11、16……で、以後5ターンごとです。</p>
        <p>各チームは生産ターンに所有拠点の空き枠を1つ選び、チーム全体で1体を生産できます。生産せずにパスすることもできます。</p>
        <div className="rules-columns">
          <div>
            <h4>直接生産できる駒</h4>
            <ul>
              <li>歩兵</li>
              <li>騎兵</li>
              <li>弓兵</li>
              <li>工</li>
              <li>忍者</li>
              <li>軍師</li>
            </ul>
          </div>
          <div>
            <h4>直接生産できない駒</h4>
            <ul>
              <li>王</li>
              <li>重歩兵</li>
            </ul>
          </div>
        </div>
      </RuleSection>

      <RuleSection number={9} title="拠点攻略と褒賞">
        <p>敵拠点の守備隊を排除すると拠点を攻略できます。攻略チームには、攻略した拠点へ褒賞駒を配置する機会があります。</p>
        <p>別チームの貢献が大きい場合は、功績補償が発生する場合があります。</p>
        <h4>王撃破</h4>
        <p>プレイヤーチームが敵王を撃破した場合、拠点の引き継ぎと王攻略褒賞が発生します。</p>
        <p>中立守備隊が敵王を撃破した場合は、次のように処理します。</p>
        <ul>
          <li>対象チームは通常どおり敗北します。</li>
          <li>対象拠点は中立守備隊の所有になります。</li>
          <li>中立守備隊への王攻略褒賞は発生しません。</li>
          <li>褒賞配置を挟まずゲームを続行します。</li>
        </ul>
      </RuleSection>

      <RuleSection number={10} title="敗北と勝利">
        <p>次のどちらかでチームは敗北します。</p>
        <ul>
          <li>王を倒される</li>
          <li>所有拠点が0になる</li>
        </ul>
        <p>敗北したチームの残存駒は盤面から除去され、以後の行動順から外れます。</p>
        <p>活動中の非中立チームが1チーム以下になった時点でゲームは終了します。1チームが残っていれば、そのチームが勝者です。</p>
      </RuleSection>
    </section>

    <section className="rules-group" aria-labelledby="unit-rules-title">
      <h2 id="unit-rules-title">ユニット・特殊能力</h2>

      <RuleSection number={11} title="ユニット一覧">
        <div className="rules-table-wrap">
          <table className="rules-table rules-unit-table">
            <thead>
              <tr><th scope="col">兵種</th><th scope="col">HP</th><th scope="col">移動力</th><th scope="col">射程</th><th scope="col">特徴</th></tr>
            </thead>
            <tbody>
              {unitRows.map(([name, hp, move, range, feature]) => <tr key={name}>
                <th scope="row">{name}</th><td>{hp}</td><td>{move}</td><td>{range}</td><td>{feature}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
        <h4>生存上限</h4>
        <ul>
          <li>忍者: 2体</li>
          <li>軍師: 2体</li>
          <li>弓兵: 基本3体。敗北済みの非中立チーム1つにつき上限が1増加</li>
        </ul>
      </RuleSection>

      <RuleSection number={12} title="軍師">
        <p>軍師には、鼓舞・転送・建設の3つの役割があります。</p>

        <h4>鼓舞軍師</h4>
        <p>範囲内の味方駒の命中率を上げます。自軍拠点から2マス以内では周囲1マス、それより離れている場合は周囲2マスが範囲です。</p>
        <p>命中率の分母が1減ります。たとえば1/7は1/6、1/6は1/5、1/5は1/4になります。</p>
        <p>複数の鼓舞は重複せず、鼓舞軍師自身は自分を鼓舞できません。</p>

        <h4>転送軍師</h4>
        <p>範囲内の味方駒を、作戦圏内の空き道路、使用可能な橋、自軍拠点の空き枠へ転送できます。</p>
        <ul>
          <li>王・工・軍師は転送できません。</li>
          <li>通常移動とは異なり、転送では対象駒と転送先を予約します。</li>
          <li>同じ対象・同じ転送先を複数の転送計画で重複使用できません。</li>
          <li>成功後に再使用できるのは5ターン後です。</li>
        </ul>

        <h4>建設軍師</h4>
        <p>橋と障害物を設置・管理します。建設軍師が死亡しても、設置済みの設備は消えず、管理者だけが不在になります。</p>
        <p>条件を満たす同じチームの後任建設軍師は、管理上限の範囲内で設備を引き継げます。設備を撤去した後、同じ種類を再設置できるのは5ターン後です。</p>
      </RuleSection>

      <RuleSection number={13} title="忍者">
        <ul>
          <li>1チームにつき最大2体です。</li>
          <li>道路から隣接する湖へ入れます。</li>
          <li>湖上では周囲8方向へ1マス移動できます。</li>
          <li><strong>湖から隣接する道路へ上陸できます。</strong></li>
          <li>湖上から橋へ直接移動することはできません。</li>
          <li>湖上では敵から隠れますが、所有者からは見えます。</li>
          <li>隠れている水上忍者同士が同じ場所へ進もうとすると、互いの存在が明らかになります。</li>
          <li>水上の敵忍者同士は攻撃できます。</li>
          <li>地上・橋上の駒と湖上忍者は、互いに直接攻撃できません。</li>
          <li>敵水上忍者の位置へ橋が建設されると、その忍者は橋上へ移され、姿を現します。</li>
          <li>自軍水上忍者がいる位置は、自軍の橋候補にできません。</li>
        </ul>
      </RuleSection>

      <RuleSection number={14} title="重歩兵">
        <p>重歩兵は、条件を満たす同じチームの通常歩兵2体を合流させて編成します。</p>
        <ul>
          <li>HP 2、移動力 1、射程 1</li>
          <li>直接生産できません。</li>
          <li>合流に使用した歩兵は、その移動フェーズで追加移動できません。</li>
        </ul>
        <h4>橋撤去による水計</h4>
        <ul>
          <li>HPを1失います。</li>
          <li>HPが残れば生存します。</li>
          <li>合法な道路、または空きのある自軍拠点へ退避します。</li>
          <li>HPが0になるか、退避先がなければ除去されます。</li>
        </ul>
      </RuleSection>

      <RuleSection number={15} title="建設・工作">
        <h4>障害物</h4>
        <ul>
          <li>作戦圏内の合法な場所へ設置できます。</li>
          <li>駒がいる場所には設置できません。</li>
          <li>移動を妨げますが、攻撃を直接遮断しません。</li>
        </ul>
        <h4>橋</h4>
        <p>作戦圏内の道路から湖方向へ直線的に探索し、条件を満たす対岸の道路がある場合に橋候補が生成されます。完成した橋は敵味方を問わず利用できます。</p>
        <p>建設軍師が死亡しても、設置済みの橋・障害物は残ります。</p>
      </RuleSection>
    </section>

    <section className="rules-group" aria-labelledby="detail-rules-title">
      <h2 id="detail-rules-title">詳細ルール</h2>

      <RuleSection number={16} title="攻撃成功率">
        <p>表内の数値は基本命中率です。</p>
        <div className="rules-table-wrap" tabIndex={0} aria-label="攻撃成功率表（横方向にスクロールできます）">
          <table className="rules-table rules-attack-table">
            <thead>
              <tr>
                <th scope="col">攻撃側＼対象</th><th scope="col">王</th><th scope="col">歩兵</th><th scope="col">重歩兵</th><th scope="col">騎兵</th><th scope="col">弓兵</th><th scope="col">忍者</th><th scope="col">工</th><th scope="col">軍師</th>
              </tr>
            </thead>
            <tbody>
              {attackRows.map(([name, ...values]) => <tr key={name}>
                <th scope="row">{name}</th>{values.map((value, index) => <td key={`${name}-${index}`}>{value}</td>)}
              </tr>)}
            </tbody>
          </table>
        </div>
        <ul className="rules-table-notes">
          <li>工は拠点内の敵だけを攻撃できます。工同士は1/6、それ以外は1/5です。</li>
          <li>軍師は攻撃できません。</li>
          <li>鼓舞を受けた駒は命中率の分母が1減ります。</li>
        </ul>
      </RuleSection>

      <RuleSection number={17} title="拠点攻略の優先順位">
        <p>複数チームが同じ拠点攻略に関与した場合、次の順で攻略チームを決めます。</p>
        <ol>
          <li>守備駒を最も多く倒したチーム</li>
          <li>生存中の最寄り駒までの経路距離が短いチーム</li>
          <li>有効な攻略ターン数が多いチーム</li>
          <li>完全同率なら抽選</li>
        </ol>
        <p>最後の有効な攻略攻撃から10ターン攻撃がない場合、攻略記録はリセットされます。</p>
      </RuleSection>

      <RuleSection number={18} title="その他の重要ルール">
        <ul>
          <li>通常移動は即時反映され、将来の移動先予約はありません。</li>
          <li>転送の対象・転送先には予約があります。</li>
          <li>障害物は移動を妨げますが、攻撃を直接遮断しません。</li>
          <li>建設軍師が死亡しても、完成済みの設備は残ります。</li>
          <li>建設軍師1体の標準管理上限は、橋1つ・障害物1つです。</li>
          <li>1チーム攻略時は指定した建設軍師1体、2チーム以上攻略時はすべての建設軍師について、種類ごとの管理上限が2つになります。</li>
          <li>中立守備隊による王撃破では、王攻略褒賞は発生しません。</li>
        </ul>
      </RuleSection>
    </section>
  </div>;
}
import type { ReactNode } from "react";
