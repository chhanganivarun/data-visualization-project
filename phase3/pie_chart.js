const arcs = pie(data);

const svg = d3
	.create("svg")
	.attr("viewBox", [-width / 2, -height / 2, width, height]);

svg.append("g")
	.attr("stroke", "white")
	.selectAll("path")
	.data(arcs)
	.join("path")
	.attr("fill", (d) => color(d.data.name))
	.attr("d", arc)
	.append("title")
	.text((d) => `${d.data.name}: ${d.data.value.toLocaleString()}`);

svg.append("g")
	.attr("font-family", "sans-serif")
	.attr("font-size", 12)
	.attr("text-anchor", "middle")
	.selectAll("text")
	.data(arcs)
	.join("text")
	.attr("transform", (d) => `translate(${arcLabel.centroid(d)})`)
	.call((text) =>
		text
			.append("tspan")
			.attr("y", "-0.4em")
			.attr("font-weight", "bold")
			.text((d) => d.data.name)
	)
	.call((text) =>
		text
			.filter((d) => d.endAngle - d.startAngle > 0.25)
			.append("tspan")
			.attr("x", 0)
			.attr("y", "0.7em")
			.attr("fill-opacity", 0.7)
			.text((d) => d.data.value.toLocaleString())
	);
